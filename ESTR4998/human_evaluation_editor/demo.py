"""Start an editor with invented review material in a temporary directory."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import secrets
import tempfile

from server import server as editor


TASK = "Suggest one way a museum could let visitors hear an object without touching it."
ANSWERS = (
    "A listening cone beside the case plays a recording of the old station clock striking noon.",
    "A small listening horn at the display plays a recording of the station clock's noon chime.",
    "A motion sensor turns the crack pattern in a clay bowl into a short melody as a visitor walks by.",
)


def write_examples(directory):
    """Write one new synthetic row per panel, with each human answer left blank."""
    responses = [{"id": f"x{n}", "text": answer} for n, answer in enumerate(ANSWERS, 1)]
    packet = json.dumps({"task": TASK, "responses": responses}, sort_keys=True).encode()
    rows = {
        "grouping": (editor.GROUPING_FIELDS, {
            "item": "display-1", "task": TASK, "responses": json.dumps(responses),
            "packet_sha256": hashlib.sha256(packet).hexdigest(),
            "groups": "{}", "status": "draft",
        }),
        "pair": (editor.PAIR_FIELDS, {
            "pair_id": "pair-1", "prompt": TASK,
            "response_a": ANSWERS[0], "response_b": ANSWERS[1],
        }),
        "quality": (editor.QUALITY_FIELDS, {
            "item_id": "answer-1", "prompt": TASK, "response": ANSWERS[2],
        }),
        "calibration": (editor.CALIBRATION_FIELDS, {
            "item_id": "question-1", "kind": "question",
            "prompt": "Two exhibits use the same bell recording but use different headphones. What would you count as a change in the central idea?",
        }),
        "inspection": (editor.INSPECTION_FIELDS, {
            "item": "reading-1", "measure": "functional_classes", "view": "illustration",
            "score": "2", "prompt_id": "museum", "heading": "Illustrative grouping",
            "score_line": "Illustrative count supplied with the demo: 2", "prompt": TASK,
            "responses": json.dumps([{"label": r["id"], "text": r["text"],
                                      "detail": "Clock recording" if i < 2 else "Bowl melody"}
                                     for i, r in enumerate(responses)]),
        }),
        "fixture": (editor.FIXTURE_FIELDS, {
            "item": "paraphrase-1", "kind": "response", "set_id": "display",
            "task_id": "museum", "status": "current", "heading": "Check a paraphrase",
            "task": TASK, "context": json.dumps([{"label": "Original", "text": ANSWERS[0]}]),
            "texts": json.dumps([{"label": "Paraphrase", "text": ANSWERS[1]}]),
            "question_1": "Do the two exhibit designs preserve the same central idea?",
        }),
    }
    paths = {}
    for kind, (fields, row) in rows.items():
        path = directory / f"{kind}.csv"
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        paths[kind] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater", choices=("A", "B"), default="A")
    parser.add_argument("--port", type=int, help="default 8081 for A, 8082 for B")
    args = parser.parse_args()
    port = args.port if args.port is not None else {"A": 8081, "B": 8082}[args.rater]
    with tempfile.TemporaryDirectory(prefix="rating-example-") as directory:
        root = Path(directory)
        paths = write_examples(root)
        password = secrets.token_urlsafe(16)
        sign_in = root / "session-password"
        sign_in.write_text(password, encoding="utf-8")
        print(f"Temporary sign-in password for Rater {args.rater}: {password}", flush=True)
        options = ["--label", f"Rater {args.rater}", "--port", str(port),
                   "--password-file", str(sign_in), "--backup-dir", str(root / "backups")]
        for panel, path in paths.items():
            options.extend([f"--{panel}-csv", str(path)])
        try:
            editor.main(options)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
