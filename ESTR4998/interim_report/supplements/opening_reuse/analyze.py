"""Post hoc opening reuse v1. Extract counts from frozen API development runs or replay them."""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from itertools import product
from math import comb
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
API_STUDY = HERE.parents[2] / "api_study"
sys.path.insert(0, str(API_STUDY))
from factorial_analysis import _interval

TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-]*|\d+")
TASKS = tuple(sorted(("curated-0", "curated-2", "curated-6", "curated-7", "curated-8",
                      "curated-10", "curated-16", "curated-18", "narrative-dev-01",
                      "narrative-dev-02")))
DEFAULT, NAIVE, CAUSAL, ALTERNATIVE = ("default", "naive_creative", "manual_causal_shift_v1",
                                      "manual_alternative_selection_v1")
TEMPERATURES = {"0.8": "t0p8", "1.0": "t1p0", "1.2": "t1p2"}
SOURCES = {
    "A": ("main-20261115-exploratory-v1",
          "9989896b9d3ffd279e0db09dbbe7e63aef31b74b0e15e14ff00f1ba70a5747b9"),
    "C": ("factorial-20261115-v1",
          "b7995e21fbccce8b526c8b0712d9c34685c576c30f0b615693b4a838a45401ea"),
    "D": ("second-generator-factorial-20261116-v1",
          "36206afa2823330dc4486a0d4ad0f9166882b21492ae3a128e0c33da0260a12c"),
}
KEY = ("study", "arm", "task", "repetition")
COUNTS = ("responses", "short_nonempty_openings", "empty_openings", "matching_pairs", "planned_pairs")
FIELDS = KEY + COUNTS


def arm_ids(study):
    if study == "A":
        return (DEFAULT, NAIVE, CAUSAL, ALTERNATIVE)
    return tuple(f"{arm}-{temp}" for arm, temp in product((DEFAULT, NAIVE, CAUSAL), TEMPERATURES.values()))


def set_counts(texts):
    """Each unordered pair can contribute one match. Empty token sequences never match."""
    if len(texts) != 10 or any(not isinstance(text, str) for text in texts):
        raise ValueError("Each complete set needs ten observed response strings")
    openings = [tuple(token.lower() for token in TOKEN.findall(text)[:8]) for text in texts]
    counts = Counter(opening for opening in openings if opening)
    return {
        "responses": len(texts),
        "short_nonempty_openings": sum(0 < len(opening) < 8 for opening in openings),
        "empty_openings": openings.count(()),
        "matching_pairs": sum(comb(n, 2) for n in counts.values()),
        "planned_pairs": comb(len(texts), 2),
    }


def extract():
    """Check exact frozen file identities before accessing their successful generation rows."""
    rows = []
    for study, (run, expected) in SOURCES.items():
        raw = (API_STUDY / "runs" / run / "results.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Unexpected {study} result file hash")
        data = json.loads(raw)
        if data["development_only"] is not True or data["n"] != 10:
            raise ValueError(f"Unexpected {study} study scope")
        for item in data["rows"]:
            cell, generations = item["cell"], item["generations"]
            if (item["generation_complete"] is not True or item["samples"] != 10
                    or any(g is None or g["status"] != "ok" for g in generations)):
                raise ValueError(f"Missing or failed response in {study}, {cell}")
            rows.append({"study": study, **{key: cell[key] for key in KEY[1:]},
                         **set_counts([g["text"] for g in generations])})
    return sorted(rows, key=lambda row: tuple(row[key] for key in KEY))


def checked_rows(rows):
    """Require the exact task, condition and repetition roster before aggregating."""
    roster = {(study, arm, task, rep) for study in SOURCES
              for arm, task, rep in product(arm_ids(study), TASKS, (0, 1))}
    result = {}
    for row in rows:
        key = (row["study"], row["arm"], row["task"], int(row["repetition"]))
        counts = {field: int(row[field]) for field in COUNTS}
        valid = 10 - counts["empty_openings"]
        if (key not in roster or key in result or counts["responses"] != 10
                or counts["planned_pairs"] != 45 or not 0 <= valid <= 10
                or not 0 <= counts["short_nonempty_openings"] <= valid
                or not 0 <= counts["matching_pairs"] <= comb(valid, 2)):
            raise ValueError(f"Unexpected or duplicate set or invalid counts at {key}")
        result[key] = counts
    if result.keys() != roster:
        raise ValueError(f"Missing {len(roster - result.keys())} planned sets")
    return result


def task_rates(sets, study, arm):
    """Each of two ten-response sets contributes 45 within-set pairs per task."""
    return [sum(sets[study, arm, task, rep]["matching_pairs"] for rep in (0, 1)) / 90
            for task in TASKS]


def report(sets):
    print("# Opening reuse in the saved API development responses\n")
    print("Post hoc eight-token surface measure, version 1. "
          "Each condition covers ten tasks, 20 sets, 200 responses and 900 within-set pairs. "
          "See the [method and interpretation](README.md).\n")
    print("| Study | Condition | Pairs with a shared opening / 900 | Equal-task mean (%) |")
    print("|---|---|---:|---:|")
    for study in SOURCES:
        for arm in arm_ids(study):
            matches = sum(sets[study, arm, task, rep]["matching_pairs"]
                          for task, rep in product(TASKS, (0, 1)))
            print(f"| {study} | `{arm}` | {matches} | {100 * mean(task_rates(sets, study, arm)):.2f} |")
    print("\n| Study | Temperature | First minus second | Difference (pp) | "
          "Nominal 95% paired task-bootstrap interval (pp) |")
    print("|---|---:|---|---:|---:|")
    for study in ("C", "D"):
        for temp, suffix in TEMPERATURES.items():
            for first, second, label in ((NAIVE, DEFAULT, "Naive − default"),
                                         (CAUSAL, NAIVE, "Causal shift − naive")):
                left = task_rates(sets, study, f"{first}-{suffix}")
                right = task_rates(sets, study, f"{second}-{suffix}")
                diffs = [a - b for a, b in zip(left, right)]
                lo, hi = _interval(diffs, TASKS, 5000, 20261116)
                print(f"| {study} | {temp} | {label} | {100 * mean(diffs):+.2f} | "
                      f"[{100 * lo:+.2f}, {100 * hi:+.2f}] |")
    short = sum(row["short_nonempty_openings"] for row in sets.values())
    empty = sum(row["empty_openings"] for row in sets.values())
    print(f"\nAll 440 planned sets had complete responses. Among 4,400 responses, {short} had "
          f"one to seven recognized tokens and {empty} had zero.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true",
                        help="print the set CSV from the three hash-checked local results files")
    args = parser.parse_args()
    if args.extract:
        rows = extract()
        checked_rows(rows)
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    else:
        with (HERE / "sets.csv").open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("Unexpected set CSV schema")
            report(checked_rows(reader))


if __name__ == "__main__":
    main()
