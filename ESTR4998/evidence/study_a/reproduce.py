"""Replay the four-arm development comparison from numeric response-set summaries."""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "api_study"))
from api_report import interval, noninferiority

ARMS = ("default", "naive_creative", "manual_causal_shift_v1", "manual_alternative_selection_v1")
TASKS = tuple(f"T{i:02d}" for i in range(1, 11))
SEED = 20261114


def read_rows(name):
    with (HERE / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def check_match(value, target):
    if not math.isclose(value, target, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Result differs from the archived reference: {value} vs {target}")


def main():
    reference = json.loads((HERE / "reference.json").read_text())
    cells = {}
    saturation, top_scores, passed, applicable = Counter(), Counter(), Counter(), Counter()
    for entry in read_rows("cells.csv"):
        arm, task = entry.pop("arm"), entry.pop("task")
        values = {k: int(v) for k, v in entry.items()}
        key = arm, task, values["repetition"]
        if key in cells or values["responses"] != 10:
            raise ValueError(f"Invalid response set {key}")
        denom = values["constraints_applicable"]
        cells[key] = {
            "distinct_count": values["distinct_count"],
            "quality": values["quality_score_sum"] / 10,
            "constraint_satisfaction": values["constraints_passed"] / denom if denom else None,
        }
        saturation[arm] += values["distinct_count"] == 10
        top_scores[arm] += values["quality_10"]
        passed[arm] += values["constraints_passed"]
        applicable[arm] += denom
    if set(cells) != {(a, t, r) for a in ARMS for t in TASKS for r in (0, 1)}:
        raise ValueError("Expected four arms, ten tasks and two repetitions")
    for arm in ARMS:
        target = reference["arms"][arm]
        summaries = {}
        for metric in ("distinct_count", "quality", "constraint_satisfaction"):
            differences, avgs = [], []
            for task in TASKS:
                observed = [cells[arm, task, r][metric] for r in (0, 1)]
                base = [cells["default", task, r][metric] for r in (0, 1)]
                if all(v is None for v in observed + base):
                    continue
                if any(v is None for v in observed + base):
                    raise ValueError(f"Incomplete block for {task}, {arm}, {metric}")
                avgs.append(mean(observed))
                differences.append(mean(observed) - mean(base))
            expected_tasks = 6 if metric == "constraint_satisfaction" else 10
            if len(avgs) != expected_tasks:
                raise ValueError(f"Expected {expected_tasks} eligible tasks for {metric}")
            if metric in target:
                check_match(mean(avgs), target[metric])
            if arm == "default":
                continue
            bounds = interval(differences, f"{SEED}-anchor-{arm}-{metric}")
            contrast = target[f"{metric}_contrast"]
            check_match(mean(differences), contrast["delta"])
            for actual, expected in zip(bounds, contrast["interval"]):
                check_match(actual, expected)
            if "guard" in contrast:
                margin = 0.05 if metric == "constraint_satisfaction" else 0.5
                verdict = noninferiority(bounds, margin)
                if verdict != contrast["guard"]:
                    raise ValueError(f"Unexpected {metric} verdict for {arm}")
                summaries[metric] = verdict
            else:
                summaries[metric] = f'{mean(differences):+.2f} [{bounds[0]:.2f}, {bounds[1]:.2f}]'
        print(f"{arm}: groups {target['distinct_count']:.2f}, quality {target['quality']:.3f}, "
              f"group ceiling {saturation[arm]}/20, constraints {passed[arm]}/{applicable[arm]}")
        if summaries:
            print(f"  difference {summaries['distinct_count']}, quality guard {summaries['quality']}, "
                  f"constraint guard {summaries['constraint_satisfaction']}")
    charges = read_rows("cost_by_arm_role.csv")
    if {(r["arm"], r["role"]) for r in charges} != {(a, r) for a in ARMS for r in ("generation", "grouping", "quality")} or len(charges) != 12:
        raise ValueError("Incomplete cost allocation")
    attempts = sum(int(r["attempts"]) for r in charges)
    cost = sum(Decimal(r["known_usd"]) for r in charges)
    if attempts != reference["attempts"] or cost != Decimal(str(reference["known_usd"])) or any(int(r["unknown_cost_attempts"]) for r in charges):
        raise ValueError("Archived accounting totals disagree")
    if sum(top_scores.values()) != 530:
        raise ValueError("Archived score ceiling disagrees")
    print(f"Quality ceiling {sum(top_scores.values())}/800. Allocated submitted requests {attempts}, USD {cost}.")


if __name__ == "__main__":
    main()
