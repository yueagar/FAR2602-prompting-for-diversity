#!/usr/bin/env python3
"""Password-protected editor for the human-evaluation sheets. Python standard library only.

Each sheet is optional: --fixture-csv edits a judge-fixture checking sheet at
/fixtures, --pair-csv a pair sheet at /pair, --quality-csv a quality sheet at
/quality, --calibration-csv a calibration sheet at /calibration, and
--inspection-csv a manual-inspection sheet at /inspection. --grouping-csv adds
the independent fresh-response grouping sheet at /grouping. Every page shows a
navigation bar to the other available sheets, except those in NAV_HIDDEN.
"""

import argparse
import contextlib
import csv
import fcntl
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import ssl
import sys
import tempfile
import threading
import time
from html import escape
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
# Each sheet type keeps its page, script and styles (and its local sheets) in
# its own folder beside server/: pair/, quality/, calibration/, inspection/,
# fixtures/.
EDITOR = ROOT.parent
# Shared pure schema validation; importing it reads no rater files.
if str(EDITOR / "grouping") not in sys.path:
    sys.path.insert(0, str(EDITOR / "grouping"))
import grouping_schema as grouping_data

TEMPLATE_PAIR_FILENAME = "pairs.csv"
DEFAULT_LABEL = "Rater A"
# This instance's settings. main() fills them in from the command line, so
# importing this module reads no files.
CONFIG = SimpleNamespace(
    pair_csv=None,
    password=None,
    label=DEFAULT_LABEL,
    quality_csv=None,
    calibration_csv=None,
    inspection_csv=None,
    fixture_csv=None,
    grouping_csv=None,
    session_cookie="pair_session_8081",
    login_cookie="pair_login_8081",
    # Every changed save first copies the previous sheet here. Tests point it
    # at a temporary folder so they never write beside the raters' backups.
    backup_dir=ROOT / "backups",
    # Set when the server itself speaks TLS (--tls-cert/--tls-key): cookies
    # then carry the Secure attribute.
    secure_cookies=False,
)
LOCK = threading.RLock()
SESSIONS = {}
# Sign-in page tokens are signed rather than stored, so requesting /login many
# times cannot evict a token another person is about to use. A token is
# "<expiry>.<nonce>.<HMAC>" under a per-process key: a restart invalidates them.
LOGIN_TOKEN_KEY = secrets.token_bytes(32)
LOGIN_TOKEN_LIFETIME = 600
# Tokens already used for a successful sign-in, until they expire. Only a
# correct password adds one, so this cannot be flooded.
USED_LOGIN_TOKENS = {}
# Serialise password checks. A failed sign-in waits briefly before releasing the gate.
LOGIN_GATE = threading.Lock()
FAILED_LOGIN_DELAY = 1.0
LOGIN_QUEUE_TIMEOUT = 30.0
PAIR_FIELDS = [
    "pair_id",
    "prompt",
    "response_a",
    "response_b",
    "equivalent",
    "rater_notes",
]
SCORES = ["relevance", "coherence", "constraints", "effectiveness"]
QUALITY_FIELDS = ["item_id", "prompt", "response", *SCORES, "rater_notes"]
# Calibration rows are either a round-1 pair to explain ("pair") or a question
# about the instructions ("question"); the page shows the fields for its kind.
CALIBRATION_FIELDS = [
    "item_id", "kind", "prompt", "response_a", "response_b", "your_round1_label",
    "proposed", "content_a", "content_b", "same_content", "style_affected",
    "answer", "rater_notes",
]
CALIBRATION_CHOICES = {"same_content": ("", "yes", "no", "unsure"),
                       "style_affected": ("", "yes", "no", "unsure")}
CALIBRATION_TEXT = ["content_a", "content_b", "answer", "rater_notes"]
CALIBRATION_EDITABLE = [*CALIBRATION_CHOICES, *CALIBRATION_TEXT]
# Manual-inspection rows (review interface): the item's metadata,
# linked by an item ID;
# the texts to read; and this reader's two answers. `responses` is a JSON list
# of {"label", "detail", "text"}; `score_line`, `side` and each response's
# `detail` hold the instrument's verdict, which the page hides until asked.
INSPECTION_META = [
    "item", "measure", "view", "side", "score", "condition", "prompt_id",
    "replicate", "generation_index",
]
INSPECTION_FIELDS = [
    *INSPECTION_META, "heading", "score_line", "prompt", "responses", "agrees", "note",
]
INSPECTION_CHOICES = ("", "yes", "no", "unsure")
INSPECTION_EDITABLE = ["agrees", "note"]
# Judge-fixture checking rows (synthetic demonstration),
# with blinded evidence and editable decisions. `kind` is "outlines" (a grouping set's
# outlines), "response" (one grouping response) or "pair" (a quality pair).
# `context` and `texts` are JSON lists of {"label", "text"}: what to compare
# against, and what is being checked. Each row carries its own question wording;
# `question_2` is blank when a row has one question, and `answer_2` must then
# stay blank. `status` is "current" or "superseded" (no longer in the latest
# fixture file, kept as a record).
FIXTURE_META = ["item", "kind", "set_id", "task_id", "assignment", "fixture_file", "status"]
FIXTURE_FIELDS = [
    *FIXTURE_META, "heading", "task", "context", "texts",
    "question_1", "question_2", "answer_1", "answer_2", "note",
]
FIXTURE_KINDS = ("outlines", "response", "pair")
FIXTURE_STATUSES = ("current", "superseded")
FIXTURE_CHOICES = ("", "yes", "no", "unsure")
FIXTURE_EDITABLE = ["answer_1", "answer_2", "note"]
GROUPING_FIELDS = grouping_data.GROUPING_FIELDS
GROUPING_EDITABLE = grouping_data.GROUPING_EDITABLE
# Static files: URL -> (page whose sheet must be available, or None for always; file).
ASSETS = {
    "/style.css": (None, ROOT / "style.css"),
    "/pair.js": ("/pair", EDITOR / "pair" / "pair.js"),
    **{
        f"/{kind}.{ext}": (f"/{kind}", EDITOR / kind / f"{kind}.{ext}")
        for kind in ("quality", "calibration", "inspection", "fixtures", "grouping")
        for ext in ("css", "js")
    },
}
# Every page links to the other available sheets, in this order.
PAGES = (
    ("/grouping", "Fresh-response grouping"),
    ("/fixtures", "Fixture checks"),
    ("/pair", "Pair ratings"),
    ("/quality", "Quality ratings"),
    ("/calibration", "Calibration"),
    ("/inspection", "Manual inspection"),
)
# Show all available demonstration panels.
NAV_HIDDEN = set()

# Labels and CSV filenames are pasted into HTML, JavaScript strings and a
# Content-Disposition header, so they are limited to characters safe in all three.
SAFE_NAME = re.compile(r"[\w .()+-]{1,100}", re.ASCII)


def allowed_origin(origin, host):
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            return False
        authority = (
            parsed.hostname.lower(),
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
        requested = urlsplit(f"{parsed.scheme}://{host or ''}")
        return authority == (
            requested.hostname,
            requested.port or (443 if parsed.scheme == "https" else 80),
        )
    except ValueError:
        return False


def available_sheets():
    """Check configured files on every request, including after removal."""
    return (
        CONFIG.pair_csv is not None and CONFIG.pair_csv.is_file(),
        CONFIG.quality_csv is not None and CONFIG.quality_csv.is_file(),
    )


def calibration_available():
    return CONFIG.calibration_csv is not None and CONFIG.calibration_csv.is_file()


def inspection_available():
    return CONFIG.inspection_csv is not None and CONFIG.inspection_csv.is_file()


def fixtures_available():
    return CONFIG.fixture_csv is not None and CONFIG.fixture_csv.is_file()


def grouping_available():
    return CONFIG.grouping_csv is not None and CONFIG.grouping_csv.is_file()


def grouping_configured():
    """Keep the grouping page available while its fresh-response sheet is pending."""
    return CONFIG.grouping_csv is not None


def availability():
    """Available pages, including configured grouping before its data is loaded."""
    pair, quality = available_sheets()
    return {
        "/grouping": grouping_configured(),
        "/fixtures": fixtures_available(),
        "/pair": pair,
        "/quality": quality,
        "/calibration": calibration_available(),
        "/inspection": inspection_available(),
    }


def landing_route():
    """Prefer fresh grouping, then fixtures and the existing rating sheets."""
    available = availability()
    for route in ("/grouping", "/fixtures", "/calibration", "/quality", "/pair", "/inspection"):
        if available[route]:
            return route
    return None


def navigation(current):
    """Links to every available listed sheet, the current one marked; empty if
    it is the only one."""
    available = availability()
    pages = [(route, title) for route, title in PAGES
             if available[route] and route not in NAV_HIDDEN]
    if len(pages) < 2:
        return ""
    current_mark = ' aria-current="page"'
    links = "".join(
        f'<a href="{route}"{current_mark if route == current else ""}>{title}</a>'
        for route, title in pages
    )
    return f'<nav class="sheets" aria-label="Sheets">{links}</nav>'


def localised():
    """Whether the pages, written for Rater A, need this instance's CSV filename or label."""
    return (
        CONFIG.label != DEFAULT_LABEL
        or (CONFIG.pair_csv is not None and CONFIG.pair_csv.name != TEMPLATE_PAIR_FILENAME)
    )


def localise(text):
    """Replace pairs.csv with the CSV filename, then Rater A with the label.

    The second replacement leaves the filenames inserted by the first alone."""
    parts = text.split(TEMPLATE_PAIR_FILENAME)
    filename = CONFIG.pair_csv.name if CONFIG.pair_csv is not None else TEMPLATE_PAIR_FILENAME
    return filename.join(
        part.replace(DEFAULT_LABEL, CONFIG.label) for part in parts
    )


def quality_page():
    page = (EDITOR / "quality" / "quality.html").read_text(encoding="utf-8")
    return page.replace("{{label}}", escape(CONFIG.label)).replace(
        "{{quality_csv}}", escape(CONFIG.quality_csv.name)
    ).replace("{{nav}}", navigation("/quality"))


def calibration_page():
    page = (EDITOR / "calibration" / "calibration.html").read_text(encoding="utf-8")
    return page.replace("{{label}}", escape(CONFIG.label)).replace(
        "{{calibration_csv}}", escape(CONFIG.calibration_csv.name)
    ).replace("{{nav}}", navigation("/calibration"))


def inspection_page():
    page = (EDITOR / "inspection" / "inspection.html").read_text(encoding="utf-8")
    return page.replace("{{label}}", escape(CONFIG.label)).replace(
        "{{inspection_csv}}", escape(CONFIG.inspection_csv.name)
    ).replace("{{nav}}", navigation("/inspection"))


def fixtures_page():
    page = (EDITOR / "fixtures" / "fixtures.html").read_text(encoding="utf-8")
    return page.replace("{{label}}", escape(CONFIG.label)).replace(
        "{{fixture_csv}}", escape(CONFIG.fixture_csv.name)
    ).replace("{{nav}}", navigation("/fixtures"))


def grouping_page():
    page = (EDITOR / "grouping" / "grouping.html").read_text(encoding="utf-8")
    return page.replace("{{label}}", escape(CONFIG.label)).replace(
        "{{grouping_csv}}", escape(CONFIG.grouping_csv.name)
    ).replace("{{nav}}", navigation("/grouping"))


def pair_page():
    page = (EDITOR / "pair" / "pair.html").read_text(encoding="utf-8")
    if localised():
        page = localise(page)
    return page.replace("{{nav}}", navigation("/pair"))


def read_sheet(path, fields, key, kind):
    """Read a sheet, refusing anything a save could not write back unchanged."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError("Unexpected CSV columns; refused to modify the sheet")
        rows = list(reader)
    # A row with more or fewer cells than the header would be rewritten with
    # its cells shifted or dropped (and extra cells break the JSON versions).
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(
            "A row does not match the CSV columns; refused to modify the sheet"
        )
    if len({row[key] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate {kind} IDs")
    return rows


def read_rows():
    return read_sheet(CONFIG.pair_csv, PAIR_FIELDS, "pair_id", "pair")


def read_items():
    return read_sheet(CONFIG.quality_csv, QUALITY_FIELDS, "item_id", "item")


def read_calibration():
    return read_sheet(CONFIG.calibration_csv, CALIBRATION_FIELDS, "item_id", "item")


def read_inspection():
    rows = read_sheet(CONFIG.inspection_csv, INSPECTION_FIELDS, "item", "item")
    for row in rows:
        try:
            responses = json.loads(row["responses"])
        except json.JSONDecodeError:
            responses = None
        if not (isinstance(responses, list) and responses and all(
            isinstance(entry, dict) and isinstance(entry.get("text"), str)
            for entry in responses
        )):
            raise ValueError(f"Item {row['item']} has no readable responses")
    return rows


def labelled_texts(value):
    """A JSON list of {"label": str, "text": str}, or None if it is not one."""
    try:
        entries = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(entries, list) and all(
        isinstance(entry, dict) and isinstance(entry.get("label"), str)
        and isinstance(entry.get("text"), str) for entry in entries
    ):
        return entries
    return None


def read_fixtures():
    rows = read_sheet(CONFIG.fixture_csv, FIXTURE_FIELDS, "item", "item")
    for row in rows:
        texts, context = labelled_texts(row["texts"]), labelled_texts(row["context"])
        if not texts or context is None:
            raise ValueError(f"Item {row['item']} has no readable texts")
        if row["kind"] not in FIXTURE_KINDS or row["status"] not in FIXTURE_STATUSES:
            raise ValueError(f"Item {row['item']} has an unknown kind or status")
        if not row["question_1"]:
            raise ValueError(f"Item {row['item']} has no question")
    return rows


def read_grouping():
    try:
        rows = read_sheet(CONFIG.grouping_csv, GROUPING_FIELDS, "item", "item")
    except FileNotFoundError:
        # A pending import is a waiting page, not a fabricated blank CSV. Other
        # read/validation errors still surface instead of looking like no data.
        return []
    for row in rows:
        grouping_data.validate_row(row)
    # A sheet must never combine answers from different blinded packets.
    if len({row["packet_sha256"] for row in rows}) > 1:
        raise ValueError("Grouping sheet mixes different response packets")
    return rows


def version(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def backup(path):
    """Copy path into CONFIG.backup_dir, readable only by its owner like the sheets."""
    backups = Path(CONFIG.backup_dir)
    backups.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = backups / f"{path.stem}-{time.time_ns()}{path.suffix}"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as destination, path.open("rb") as source:
        shutil.copyfileobj(source, destination)
    return target


def replace_csv(path, fields, rows):
    """Copy the current file into the backup folder, then replace it atomically."""
    backup(path)
    fd, name = tempfile.mkstemp(prefix=".rating-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_rating(data):
    if set(data) != {"pair_id", "equivalent", "rater_notes", "version"}:
        raise ValueError("Only the verdict and notes can be edited")
    if not all(isinstance(value, str) for value in data.values()):
        raise ValueError("All fields must be text")
    if data["equivalent"] not in ("", "yes", "no"):
        raise ValueError("Verdict must be yes, no, or blank")
    if len(data["rater_notes"]) > 10000:
        raise ValueError("Notes must be at most 10,000 characters")
    with LOCK:
        rows = read_rows()
        row = next((item for item in rows if item["pair_id"] == data["pair_id"]), None)
        if row is None:
            raise ValueError("Unknown pair ID")
        if not hmac.compare_digest(version(row), data["version"]):
            return None
        if (row["equivalent"], row["rater_notes"]) == (
            data["equivalent"],
            data["rater_notes"],
        ):
            return version(row)
        row["equivalent"], row["rater_notes"] = data["equivalent"], data["rater_notes"]
        replace_csv(CONFIG.pair_csv, PAIR_FIELDS, rows)
        return version(row)


def save_score(data):
    editable = [*SCORES, "rater_notes"]
    if set(data) != {"item_id", *editable, "version"}:
        raise ValueError("Only the four scores and notes can be edited")
    if not all(isinstance(value, str) for value in data.values()):
        raise ValueError("All fields must be text")
    for field in SCORES:
        if data[field] not in ("", "1", "2", "3", "4", "5"):
            raise ValueError(
                f"{field.capitalize()} must be a whole number from 1 to 5, or blank"
            )
    if len(data["rater_notes"]) > 10000:
        raise ValueError("Notes must be at most 10,000 characters")
    with LOCK:
        rows = read_items()
        row = next((item for item in rows if item["item_id"] == data["item_id"]), None)
        if row is None:
            raise ValueError("Unknown item ID")
        if not hmac.compare_digest(version(row), data["version"]):
            return None
        if all(row[field] == data[field] for field in editable):
            return version(row)
        row.update({field: data[field] for field in editable})
        replace_csv(CONFIG.quality_csv, QUALITY_FIELDS, rows)
        return version(row)


def save_calibration(data):
    if set(data) != {"item_id", *CALIBRATION_EDITABLE, "version"}:
        raise ValueError("Only your answers and notes can be edited")
    if not all(isinstance(value, str) for value in data.values()):
        raise ValueError("All fields must be text")
    for field, allowed in CALIBRATION_CHOICES.items():
        if data[field] not in allowed:
            raise ValueError(f"{field} must be yes, no, unsure, or blank")
    for field in CALIBRATION_TEXT:
        if len(data[field]) > 10000:
            raise ValueError("Each answer must be at most 10,000 characters")
    with LOCK:
        rows = read_calibration()
        row = next((item for item in rows if item["item_id"] == data["item_id"]), None)
        if row is None:
            raise ValueError("Unknown item ID")
        if not hmac.compare_digest(version(row), data["version"]):
            return None
        if all(row[field] == data[field] for field in CALIBRATION_EDITABLE):
            return version(row)
        row.update({field: data[field] for field in CALIBRATION_EDITABLE})
        replace_csv(CONFIG.calibration_csv, CALIBRATION_FIELDS, rows)
        return version(row)


def save_inspection(data):
    if set(data) != {"item", *INSPECTION_EDITABLE, "version"}:
        raise ValueError("Only your verdict and note can be edited")
    if not all(isinstance(value, str) for value in data.values()):
        raise ValueError("All fields must be text")
    if data["agrees"] not in INSPECTION_CHOICES:
        raise ValueError("The verdict must be yes, no, unsure, or blank")
    if len(data["note"]) > 10000:
        raise ValueError("Notes must be at most 10,000 characters")
    with LOCK:
        rows = read_inspection()
        row = next((entry for entry in rows if entry["item"] == data["item"]), None)
        if row is None:
            raise ValueError("Unknown item")
        if not hmac.compare_digest(version(row), data["version"]):
            return None
        if all(row[field] == data[field] for field in INSPECTION_EDITABLE):
            return version(row)
        row.update({field: data[field] for field in INSPECTION_EDITABLE})
        replace_csv(CONFIG.inspection_csv, INSPECTION_FIELDS, rows)
        return version(row)


@contextlib.contextmanager
def sheet_lock(path):
    """Exclusive lock on a sheet's read-modify-replace, across processes.

    The editor and fixtures/fixture_sheets.py `build` both rewrite a fixture
    sheet. The thread LOCK cannot see the builder, and the sheet itself cannot
    be locked because os.replace swaps its inode, so both take this lock file
    beside it (the inspection merge uses the same pattern)."""
    path = Path(path)
    descriptor = os.open(path.parent / f".{path.name}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def save_fixture(data):
    if set(data) != {"item", *FIXTURE_EDITABLE, "version"}:
        raise ValueError("Only your answers and note can be edited")
    if not all(isinstance(value, str) for value in data.values()):
        raise ValueError("All fields must be text")
    if data["answer_1"] not in FIXTURE_CHOICES or data["answer_2"] not in FIXTURE_CHOICES:
        raise ValueError("Each answer must be yes, no, unsure, or blank")
    if len(data["note"]) > 10000:
        raise ValueError("Notes must be at most 10,000 characters")
    with LOCK, sheet_lock(CONFIG.fixture_csv):
        rows = read_fixtures()
        row = next((entry for entry in rows if entry["item"] == data["item"]), None)
        if row is None:
            raise ValueError("Unknown item")
        if not row["question_2"] and data["answer_2"]:
            raise ValueError("This item has only one question")
        if not hmac.compare_digest(version(row), data["version"]):
            return None
        if all(row[field] == data[field] for field in FIXTURE_EDITABLE):
            return version(row)
        row.update({field: data[field] for field in FIXTURE_EDITABLE})
        replace_csv(CONFIG.fixture_csv, FIXTURE_FIELDS, rows)
        return version(row)


def save_grouping(data):
    if not isinstance(data, dict) or set(data) != {"item", *GROUPING_EDITABLE, "version"}:
        raise ValueError("Only your groups, status, exposure and note can be edited")
    if not all(isinstance(value, str) for value in data.values()):
        raise ValueError("All fields must be text")
    with LOCK, sheet_lock(CONFIG.grouping_csv):
        rows = read_grouping()
        row = next((entry for entry in rows if entry["item"] == data["item"]), None)
        if row is None:
            raise ValueError("Unknown item")
        if not hmac.compare_digest(version(row), data["version"]):
            return None
        candidate = row | {field: data[field] for field in GROUPING_EDITABLE}
        grouping_data.validate_row(candidate)
        if all(row[field] == candidate[field] for field in GROUPING_EDITABLE):
            return version(row)
        row.update(candidate)
        replace_csv(CONFIG.grouping_csv, GROUPING_FIELDS, rows)
        return version(row)


def new_login_token(now):
    payload = f"{int(now) + LOGIN_TOKEN_LIFETIME}.{secrets.token_urlsafe(16)}"
    signature = hmac.new(LOGIN_TOKEN_KEY, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def login_token_valid(token, now):
    """A signed, unexpired token not yet used for a successful sign-in. Hold LOCK."""
    payload, _, signature = token.rpartition(".")
    expected = hmac.new(LOGIN_TOKEN_KEY, payload.encode(), hashlib.sha256).hexdigest()
    if not payload or not hmac.compare_digest(signature.encode(), expected.encode()):
        return False
    try:
        expires = int(payload.split(".", 1)[0])
    except ValueError:
        return False
    for used in [key for key, stamp in USED_LOGIN_TOKENS.items() if stamp < now]:
        del USED_LOGIN_TOKENS[used]
    return expires >= now and token not in USED_LOGIN_TOKENS


def check_password(supplied):
    """Compare under the shared gate; a wrong password holds it for the delay.

    Returns True or False, or None when the queue of earlier attempts is too
    long to wait for."""
    if not LOGIN_GATE.acquire(timeout=LOGIN_QUEUE_TIMEOUT):
        return None
    try:
        if hmac.compare_digest(supplied.encode(), CONFIG.password.encode()):
            return True
        time.sleep(FAILED_LOGIN_DELAY)
        return False
    finally:
        LOGIN_GATE.release()


def cookie_attributes():
    return "; Secure" if CONFIG.secure_cookies else ""


class Handler(BaseHTTPRequestHandler):
    server_version = "PairEditor"

    def reply(
        self, status, body, content_type="application/json; charset=utf-8", headers=None
    ):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'none'",
        )
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def session(self):
        try:
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            token = (
                cookies[CONFIG.session_cookie].value
                if CONFIG.session_cookie in cookies
                else ""
            )
        except Exception:
            return None
        with LOCK:
            now = time.time()
            for key in list(SESSIONS):
                if SESSIONS[key]["expires"] < now:
                    del SESSIONS[key]
            return SESSIONS.get(token)

    def missing_sheets(self):
        return self.reply(
            503, (ROOT / "error.html").read_bytes(), "text/html; charset=utf-8"
        )

    def do_GET(self):
        try:
            return self.get()
        except (ValueError, csv.Error) as error:
            # A sheet that fails its checks (columns, ragged rows, duplicate
            # IDs) is refused with a message the page can show.
            self.log_error("Refused to read a sheet: %s", error)
            return self.reply(500, {"error": f"{error}. Ask the organiser to check the CSV."})

    def get(self):
        path = urlsplit(self.path).path
        pair, quality = available_sheets()
        calibration = calibration_available()
        inspection = inspection_available()
        fixtures = fixtures_available()
        grouping = grouping_configured()
        landing = landing_route()
        if path in ASSETS and (ASSETS[path][0] is None or availability()[ASSETS[path][0]]):
            mime = "text/css" if path.endswith("css") else "text/javascript"
            body = ASSETS[path][1].read_bytes()
            if path == "/pair.js" and localised():
                body = localise(body.decode()).encode()
            return self.reply(200, body, mime + "; charset=utf-8")
        session = self.session()
        pages = ("/grouping", "/fixtures", "/pair", "/quality", "/calibration", "/inspection")
        if path in ("/", "/login", *pages) and landing is None:
            return self.missing_sheets()
        if path == "/":
            return self.reply(
                303, "", "text/plain",
                {"Location": landing if session else "/login"}
            )
        if path in pages:
            if not session:
                return self.reply(303, "", "text/plain", {"Location": "/login"})
            if not availability()[path]:
                return self.reply(303, "", "text/plain", {"Location": landing})
            render = {
                "/grouping": grouping_page,
                "/fixtures": fixtures_page,
                "/pair": pair_page,
                "/quality": quality_page,
                "/calibration": calibration_page,
                "/inspection": inspection_page,
            }[path]
            return self.reply(200, render(), "text/html; charset=utf-8")
        if path == "/login":
            token = new_login_token(time.time())
            html = (ROOT / "login.html").read_text()
            if localised():
                html = localise(html)  # before the token goes in: it may contain "Rater A"
            html = html.replace(
                "{{rating_type}}",
                {"/grouping": "Fresh-response grouping", "/fixtures": "Fixture check", "/calibration": "Calibration", "/quality": "Quality",
                 "/pair": "Pair", "/inspection": "Inspection"}[landing],
            )
            html = html.replace("{{login_token}}", token)
            return self.reply(
                200,
                html,
                "text/html; charset=utf-8",
                {
                    "Set-Cookie": f"{CONFIG.login_cookie}={token}; HttpOnly; SameSite=Strict; Path=/login; Max-Age={LOGIN_TOKEN_LIFETIME}{cookie_attributes()}"
                },
            )
        if path == "/favicon.ico":
            return self.reply(204, b"", "image/x-icon")
        if not session:
            return self.reply(401, {"error": "Please sign in again"})
        if grouping and path == "/api/grouping":
            with LOCK:
                rows = read_grouping()
            return self.reply(200, {
                "rows": [dict(row, version=version(row)) for row in rows],
                "csrf": session["csrf"],
            })
        if grouping and path == "/download-grouping":
            # Serialize validated public columns rather than serving an arbitrary
            # file. No provenance mapping, judge labels or other rater is exposed.
            with LOCK:
                rows = read_grouping()
            if not rows:
                return self.reply(409, {"error": "Fresh responses have not been loaded yet."})
            body = io.StringIO(newline="")
            writer = csv.DictWriter(body, fieldnames=GROUPING_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            return self.reply(200, body.getvalue(), "text/csv; charset=utf-8", {
                "Content-Disposition": f'attachment; filename="{CONFIG.grouping_csv.name}"',
            })
        if pair and path == "/api/pairs":
            with LOCK:
                rows = read_rows()
            return self.reply(
                200,
                {
                    "rows": [dict(row, version=version(row)) for row in rows],
                    "csrf": session["csrf"],
                },
            )
        if pair and path == "/download":
            with LOCK:
                body = CONFIG.pair_csv.read_bytes()
            return self.reply(
                200,
                body,
                "text/csv; charset=utf-8",
                {
                    "Content-Disposition": f'attachment; filename="{CONFIG.pair_csv.name}"'
                },
            )
        if quality and path == "/api/items":
            with LOCK:
                rows = read_items()
            return self.reply(
                200,
                {
                    "rows": [dict(row, version=version(row)) for row in rows],
                    "csrf": session["csrf"],
                },
            )
        if calibration and path == "/api/calibration":
            with LOCK:
                rows = read_calibration()
            return self.reply(
                200,
                {
                    "rows": [dict(row, version=version(row)) for row in rows],
                    "csrf": session["csrf"],
                },
            )
        if calibration and path == "/download-calibration":
            with LOCK:
                body = CONFIG.calibration_csv.read_bytes()
            return self.reply(
                200,
                body,
                "text/csv; charset=utf-8",
                {
                    "Content-Disposition": f'attachment; filename="{CONFIG.calibration_csv.name}"'
                },
            )
        if fixtures and path == "/api/fixtures":
            with LOCK:
                rows = read_fixtures()
            return self.reply(
                200,
                {
                    "rows": [dict(row, version=version(row)) for row in rows],
                    "csrf": session["csrf"],
                },
            )
        if fixtures and path == "/download-fixtures":
            with LOCK:
                body = CONFIG.fixture_csv.read_bytes()
            return self.reply(
                200,
                body,
                "text/csv; charset=utf-8",
                {
                    "Content-Disposition": f'attachment; filename="{CONFIG.fixture_csv.name}"'
                },
            )
        if inspection and path == "/api/inspection":
            with LOCK:
                rows = read_inspection()
            return self.reply(
                200,
                {
                    "rows": [dict(row, version=version(row)) for row in rows],
                    "csrf": session["csrf"],
                },
            )
        if inspection and path == "/download-inspection":
            with LOCK:
                body = CONFIG.inspection_csv.read_bytes()
            return self.reply(
                200,
                body,
                "text/csv; charset=utf-8",
                {
                    "Content-Disposition": f'attachment; filename="{CONFIG.inspection_csv.name}"'
                },
            )
        if quality and path == "/download-quality":
            with LOCK:
                body = CONFIG.quality_csv.read_bytes()
            return self.reply(
                200,
                body,
                "text/csv; charset=utf-8",
                {
                    "Content-Disposition": f'attachment; filename="{CONFIG.quality_csv.name}"'
                },
            )
        return self.reply(404, {"error": "Not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64000:
                return self.reply(413, {"error": "Request too large or empty"})
            self.connection.settimeout(15)
            body = self.rfile.read(length)
            path = urlsplit(self.path).path
            origin = self.headers.get("Origin")
            pair, quality = available_sheets()
            landing = landing_route()
            save_paths = tuple(
                route for route, available in (
                    ("/api/rate", pair),
                    ("/api/score", quality),
                    ("/api/calibration", calibration_available()),
                    ("/api/inspection", inspection_available()),
                    ("/api/fixtures", fixtures_available()),
                    ("/api/grouping", grouping_configured()),
                )
                if available
            )
            # Opaque origins are accepted only on routes which independently
            # verify an unpredictable cookie-bound login token or save token.
            opaque_with_token_check = origin == "null" and path in (
                "/login",
                *save_paths,
            )
            if not opaque_with_token_check and not allowed_origin(
                origin, self.headers.get("Host")
            ):
                self.log_error(
                    "Rejected origin %r for Host %r", origin, self.headers.get("Host")
                )
                return self.reply(403, {"error": "Invalid origin"})
            if path == "/login":
                if landing is None:
                    return self.missing_sheets()
                form = parse_qs(body.decode())
                submitted_token = form.get("login_token", [""])[0]
                try:
                    cookies = SimpleCookie(self.headers.get("Cookie", ""))
                    cookie_token = (
                        cookies[CONFIG.login_cookie].value
                        if CONFIG.login_cookie in cookies
                        else ""
                    )
                except Exception:
                    cookie_token = ""
                with LOCK:
                    token_ok = (
                        submitted_token
                        and hmac.compare_digest(submitted_token.encode(), cookie_token.encode())
                        and login_token_valid(cookie_token, time.time())
                    )
                if not token_ok:
                    return self.reply(
                        403,
                        "Login page expired. Reload the sign-in page and try again.",
                        "text/plain",
                    )
                matched = check_password(form.get("password", [""])[0])
                if matched is None:
                    return self.reply(
                        429,
                        "Sign-in is busy. Wait a minute, then try again.",
                        "text/plain",
                    )
                if not matched:
                    return self.reply(
                        401,
                        "Incorrect password. Go back and try again.",
                        "text/plain",
                    )
                with LOCK:
                    now = time.time()
                    if not login_token_valid(cookie_token, now):
                        return self.reply(
                            403,
                            "Login page expired. Reload the sign-in page and try again.",
                            "text/plain",
                        )
                    USED_LOGIN_TOKENS[cookie_token] = now + LOGIN_TOKEN_LIFETIME
                    token = secrets.token_urlsafe(32)
                    SESSIONS[token] = {
                        "csrf": secrets.token_urlsafe(32),
                        "expires": now + 43200,
                    }
                return self.reply(
                    303,
                    "",
                    "text/plain",
                    {
                        "Location": landing,
                        "Set-Cookie": f"{CONFIG.session_cookie}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200{cookie_attributes()}",
                    },
                )
            session = self.session()
            if not session:
                return self.reply(401, {"error": "Please sign in again"})
            if not hmac.compare_digest(
                self.headers.get("X-CSRF-Token", ""), session["csrf"]
            ):
                return self.reply(403, {"error": "Invalid save token"})
            if path not in save_paths:
                return self.reply(404, {"error": "Not found"})
            if path == "/api/grouping":
                if not grouping_available():
                    return self.reply(409, {"error": "Fresh responses have not been loaded yet."})
                saved = save_grouping(grouping_data.strict_json_loads(body.decode("utf-8")))
                if saved is None:
                    return self.reply(409, {
                        "error": "This set changed on another device or was rebuilt. Copy your groups and note, then reload.",
                    })
                return self.reply(200, {"version": saved})
            if path == "/api/fixtures":
                saved = save_fixture(json.loads(body))
                if saved is None:
                    return self.reply(
                        409,
                        {
                            "error": "This item changed on another device or was rebuilt. Copy your note, then reload."
                        },
                    )
                return self.reply(200, {"version": saved})
            if path == "/api/inspection":
                saved = save_inspection(json.loads(body))
                if saved is None:
                    return self.reply(
                        409,
                        {
                            "error": "This item changed on another device. Copy your note, then reload."
                        },
                    )
                return self.reply(200, {"version": saved})
            if path == "/api/calibration":
                saved = save_calibration(json.loads(body))
                if saved is None:
                    return self.reply(
                        409,
                        {
                            "error": "This item changed on another device. Copy your answers, then reload."
                        },
                    )
                return self.reply(200, {"version": saved})
            if path == "/api/score":
                saved = save_score(json.loads(body))
                if saved is None:
                    return self.reply(
                        409,
                        {
                            "error": "This item changed on another device. Copy your notes, then reload to review the newer scores."
                        },
                    )
                return self.reply(200, {"version": saved})
            saved = save_rating(json.loads(body))
            if saved is None:
                return self.reply(
                    409,
                    {
                        "error": "This pair changed on another device. Copy your notes, then reload to review the newer rating."
                    },
                )
            return self.reply(200, {"version": saved})
        except (ValueError, TypeError, KeyError, UnicodeError) as error:
            return self.reply(400, {"error": str(error)})
        except Exception:
            self.log_error("Unable to save/read CSV")
            return self.reply(
                500,
                {
                    "error": "Unable to save. Your edits remain on screen; retry or copy your notes."
                },
            )


class EditorServer(ThreadingHTTPServer):
    """The HTTP server, optionally speaking TLS when ssl_context is set.

    The TLS handshake runs in the request's own thread, so a slow or plain-HTTP
    client cannot stall the accept loop."""

    daemon_threads = True
    ssl_context = None

    def finish_request(self, request, client_address):
        if self.ssl_context is None:
            return super().finish_request(request, client_address)
        try:
            request.settimeout(30)
            tls = self.ssl_context.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError):
            return None  # Failed handshake, e.g. plain HTTP sent to the TLS port.
        with tls:
            tls.settimeout(None)
            super().finish_request(tls, client_address)
            self.shutdown_request(tls)  # Half-close first, as for plain HTTP.


def tls_context(cert, key):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    return context


def warn(message):
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Password-protected editor for the available human-evaluation sheets."
    )
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--pair-csv", type=Path, help="pair sheet to edit at /pair (no default)"
    )
    parser.add_argument(
        "--password-file",
        type=Path,
        required=True,
        help="file holding the editor password",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="rater label shown on the pages (default: %(default)s)",
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        help="also edit this quality sheet at /quality (default: off)",
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        help="also edit this calibration sheet at /calibration (default: off)",
    )
    parser.add_argument(
        "--inspection-csv",
        type=Path,
        help="also edit this manual-inspection sheet at /inspection (default: off)",
    )
    parser.add_argument(
        "--fixture-csv",
        type=Path,
        help="also edit this judge-fixture checking sheet at /fixtures (default: off)",
    )
    parser.add_argument(
        "--grouping-csv", type=Path,
        help="edit a blinded fresh-response grouping sheet at /grouping, the preferred landing page (default: off)",
    )
    parser.add_argument(
        "--tls-cert",
        type=Path,
        help="serve HTTPS with this PEM certificate (chain); needs --tls-key. Cookies are then Secure",
    )
    parser.add_argument(
        "--tls-key", type=Path, help="PEM private key for --tls-cert"
    )
    parser.add_argument("--backup-dir", type=Path, default=ROOT / "backups")
    args = parser.parse_args(argv)
    CONFIG.backup_dir = args.backup_dir
    if (args.tls_cert is None) != (args.tls_key is None):
        parser.error("--tls-cert and --tls-key must be given together")
    context = None
    if args.tls_cert is not None:
        try:
            context = tls_context(args.tls_cert.expanduser(), args.tls_key.expanduser())
        except (OSError, ssl.SSLError) as error:
            parser.error(f"cannot load the TLS certificate or key: {error}")
    pair_csv = args.pair_csv.expanduser().absolute() if args.pair_csv else None
    quality_csv = args.quality_csv.expanduser().absolute() if args.quality_csv else None
    calibration_csv = (
        args.calibration_csv.expanduser().absolute() if args.calibration_csv else None
    )
    inspection_csv = (
        args.inspection_csv.expanduser().absolute() if args.inspection_csv else None
    )
    fixture_csv = args.fixture_csv.expanduser().absolute() if args.fixture_csv else None
    grouping_csv = args.grouping_csv.expanduser().absolute() if args.grouping_csv else None
    allowed = "may contain only letters, digits, spaces and . _ ( ) + -"
    if not SAFE_NAME.fullmatch(args.label):
        parser.error(f"--label {allowed}")
    for path in (pair_csv, quality_csv, calibration_csv, inspection_csv, fixture_csv, grouping_csv):
        if path is not None and not SAFE_NAME.fullmatch(path.name):
            parser.error(f"CSV filenames {allowed}: {path.name!r}")
    try:
        password = args.password_file.expanduser().read_text().strip()
    except OSError as error:
        parser.error(f"cannot read the password file: {error}")
    if not password:
        parser.error(f"the password file {args.password_file} is empty")
    CONFIG.pair_csv, CONFIG.password, CONFIG.label, CONFIG.quality_csv = (
        pair_csv,
        password,
        args.label,
        quality_csv,
    )
    CONFIG.calibration_csv = calibration_csv
    CONFIG.inspection_csv = inspection_csv
    CONFIG.fixture_csv = fixture_csv
    CONFIG.grouping_csv = grouping_csv
    CONFIG.secure_cookies = context is not None
    CONFIG.session_cookie, CONFIG.login_cookie = (
        f"pair_session_{args.port}",
        f"pair_login_{args.port}",
    )
    for option, path, read in (
        ("--pair-csv", pair_csv, read_rows),
        ("--quality-csv", quality_csv, read_items),
        ("--calibration-csv", calibration_csv, read_calibration),
        ("--inspection-csv", inspection_csv, read_inspection),
        ("--fixture-csv", fixture_csv, read_fixtures),
        ("--grouping-csv", grouping_csv, read_grouping),
    ):
        if path is not None and path.is_file():
            try:
                read()
            except (OSError, ValueError, csv.Error) as error:
                parser.error(f"{path}: {error}")
        elif path is not None:
            # Relative paths resolve against the current directory, not the
            # editor folder, so a typo or a launch from elsewhere lands here.
            if option == "--grouping-csv":
                warn(f"{option} {path} does not exist; /grouping shows a waiting page until fresh responses are loaded.")
            else:
                warn(f"{option} {path} does not exist; that sheet stays unavailable until the file appears.")
    server = EditorServer((args.host, args.port), Handler)
    server.ssl_context = context
    scheme = "https" if context else "http"
    print(f"Rating editor listening on {scheme}://{args.host}:{server.server_port}", flush=True)
    pair, quality = available_sheets()
    if grouping_available():
        print("Fresh-response grouping sheet at /grouping", flush=True)
    elif grouping_configured():
        print("Fresh-response grouping at /grouping; waiting for fresh responses.", flush=True)
    if fixtures_available():
        print("Fixture-checking sheet at /fixtures", flush=True)
    if pair:
        print("Pair sheet at /pair", flush=True)
    if quality:
        print("Quality sheet at /quality", flush=True)
    if calibration_available():
        print("Calibration sheet at /calibration", flush=True)
    if inspection_available():
        print("Manual inspection available at /inspection", flush=True)
    if landing_route() is None:
        print("No rating CSV files available; showing an error page.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
