"""Offline checks of the invented sheets and localhost editor."""

import csv
from http.client import HTTPResponse
from http.cookiejar import CookieJar
from http.cookies import SimpleCookie
import html.parser
import io
import json
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo


class SignInPage(html.parser.HTMLParser):
    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "input" and attrs.get("name") == "login_token":
            self.token = attrs["value"]


class MemoryConnection:
    """Byte streams for the standard-library HTTP request and response parser."""

    def __init__(self, input_bytes):
        self.input = io.BytesIO(input_bytes)
        self.output = io.BytesIO()

    def makefile(self, *args):
        return self.input

    def sendall(self, data):
        self.output.write(data)

    def settimeout(self, seconds):
        pass


class MemoryBrowser:
    def __init__(self):
        self.cookies = SimpleCookie()

    def visit(self, path, data=None, headers=None):
        body = data or b""
        request_headers = {"Host": "127.0.0.1", "Connection": "close",
                           "Content-Length": str(len(body)), **(headers or {})}
        if self.cookies:
            request_headers["Cookie"] = "; ".join(c.OutputString(attrs=[]) for c in self.cookies.values())
        method = "POST" if data is not None else "GET"
        lines = [f"{method} {path} HTTP/1.1", *(f"{k}: {v}" for k, v in request_headers.items())]
        packet = ("\r\n".join(lines) + "\r\n\r\n").encode() + body
        conn = MemoryConnection(packet)
        demo.editor.Handler(conn, ("127.0.0.1", 0), None)
        response = HTTPResponse(MemoryConnection(conn.output.getvalue()))
        response.begin()
        for cookie in response.headers.get_all("Set-Cookie", []):
            self.cookies.load(cookie)
        payload = response.read().decode()
        if response.status == 303:
            return self.visit(response.headers["Location"])
        if response.status != 200:
            raise AssertionError(f"{method} {path} returned {response.status}")
        return payload


class DemoTests(unittest.TestCase):
    def test_synthetic_grouping_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = demo.write_examples(Path(directory))
            with paths["grouping"].open(newline="") as stream:
                row = next(csv.DictReader(stream))
            schema = demo.editor.grouping_data
            schema.validate_row(row)
            with self.assertRaisesRegex(ValueError, "Assign every response"):
                schema.validate_row({**row, "status": "complete", "exposure": "no"})
            with self.assertRaisesRegex(ValueError, "prior exposure"):
                schema.validate_row({**row, "status": "complete", "groups": '{"x1":1,"x2":1,"x3":2}'})
            with self.assertRaisesRegex(ValueError, "unknown response"):
                schema.validate_row({**row, "groups": '{"missing":1}'})
            with self.assertRaises(ValueError):
                schema.validate_row({**row, "groups": '{"x1":1,"x1":2}'})
            with self.assertRaisesRegex(ValueError, "what needs discussion"):
                schema.validate_row({**row, "status": "unsure"})

    def setUp(self):
        editor = demo.editor
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        paths = demo.write_examples(root)
        self.password = secrets.token_urlsafe()
        old_config = vars(editor.CONFIG).copy()
        self.addCleanup(lambda: vars(editor.CONFIG).update(old_config))
        for panel, path in paths.items():
            setattr(editor.CONFIG, panel + "_csv", path)
        editor.CONFIG.label = "Rater B"
        editor.CONFIG.password = self.password
        editor.CONFIG.backup_dir = root / "versions"

    def exercise_editor(self, visit, base):
        login = SignInPage()
        login.feed(visit("/"))
        form = urllib.parse.urlencode({"login_token": login.token, "password": self.password}).encode()
        self.assertIn("Rater B", visit("/login", form))
        panels = {"grouping":"grouping", "pair":"pairs", "quality":"items",
                  "calibration":"calibration", "inspection":"inspection", "fixtures":"fixtures"}
        for panel, endpoint in panels.items():
            with self.subTest(panel=panel):
                page = visit("/" + panel)
                self.assertIn("SYNTHETIC DEMO", page)
                self.assertIn("Rater B", page)
                self.assertTrue(visit("/" + panel + ".js"))
                self.assertEqual(len(json.loads(visit("/api/" + endpoint))["rows"]), 1)
        data = json.loads(visit("/api/grouping"))
        row = data["rows"][0]
        update = {"item":row["item"], "version":row["version"], "status":"complete",
                  "groups":json.dumps({"x1":1, "x2":1, "x3":2}), "note":"", "exposure":"no"}
        visit("/api/grouping", json.dumps(update).encode(),
              {"X-CSRF-Token":data["csrf"], "Content-Type":"application/json", "Origin":base})
        self.assertEqual(json.loads(visit("/api/grouping"))["rows"][0]["status"], "complete")
        exported = next(csv.DictReader(io.StringIO(visit("/download-grouping"))))
        self.assertEqual(json.loads(exported["groups"]), {"x1":1, "x2":1, "x3":2})
        self.assertEqual(exported["exposure"], "no")
        with tempfile.TemporaryDirectory() as separate:
            second_paths = demo.write_examples(Path(separate))
            with second_paths["grouping"].open(newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["status"], "draft")

    def test_pages_save_reload_and_download_in_memory(self):
        self.exercise_editor(MemoryBrowser().visit, "http://127.0.0.1")

    def test_pages_save_reload_and_download_on_localhost(self):
        try:
            server = demo.editor.EditorServer(("127.0.0.1", 0), demo.editor.Handler)
        except PermissionError:
            self.skipTest("This environment denies loopback sockets")
        with server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                browser = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                                      urllib.request.HTTPCookieProcessor(CookieJar()))
                def visit(path, data=None, headers=None):
                    request = urllib.request.Request(base + path, data=data, headers=headers or {})
                    with browser.open(request, timeout=5) as response:
                        return response.read().decode()

                self.exercise_editor(visit, base)
            finally:
                server.shutdown()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
