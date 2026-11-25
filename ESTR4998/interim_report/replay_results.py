"""Verify reported group-count contrasts and summarize public numeric inputs."""
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent / "api_study"))
from factorial_analysis import _interval


def rows(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def same(label, result, expected):
    if not math.isclose(result, float(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label}: calculated {result}, expected {expected}")


def grouping_sensitivity():
    pooled = rows("study_b_pooled_task_counts.csv")
    arms = rows("study_b_grouping_by_arm.csv")
    if len({row["task"] for row in pooled}) != 10 or len(arms) != 4:
        raise ValueError("Expected ten tasks and four arms in Study B")
    for arm in arms:
        counts = [int(row[arm["arm"]]) for row in pooled]
        same(arm["arm"], mean(counts), arm["pooled_mean_k20"])
        same(arm["arm"], counts.count(20), arm["pooled_ceiling_of_10"])
    exact = sum(int(row["identical_partitions_of_20"]) for row in arms)
    print(f"Study B: {exact}/80 exact repeat partitions. Verified four pooled means and ceilings.")


def factorial_contrasts():
    cells = rows("cross_generator_task_counts.csv")
    estimates = [row for row in rows("cross_generator_contrasts.csv") if row["measure"] == "primary_count"]
    plans = {"DeepSeek": "temperature_instruction_20261116.json",
             "Nemotron": "second_generator_20261116.json"}
    for generator, filename in plans.items():
        protocol = json.loads((DATA / filename).read_text())["analysis_protocol"]
        counts = {(r["task"], r["condition"]): float(r["primary_count"])
                  for r in cells if r["generator"] == generator}
        tasks = sorted({task for task, _ in counts})
        if len(tasks) != 10 or len(counts) != 90:
            raise ValueError(f"{generator}: expected ten tasks under nine conditions")
        frozen = {r["contrast"]: r for r in estimates if r["generator"] == generator}
        if set(frozen) != {r["id"] for r in protocol["contrasts"]} or len(frozen) != 15:
            raise ValueError(f"{generator}: incomplete contrast roster")
        for contrast in protocol["contrasts"]:
            name = contrast["id"]
            reported = frozen[name]
            delta = [sum(weight * counts[task, arm] for arm, weight in contrast["weights"].items())
                     for task in tasks]
            same(name, mean(delta), reported["mean"])
            tails = [(0.025, "ci95")]
            if contrast["analysis_role"] == "primary":
                tails.append((0.005, "ci99"))
            for tail, field in tails:
                bounds = _interval(delta, tasks, protocol["bootstrap_samples"], protocol["bootstrap_seed"], tail)
                for value, endpoint in zip(bounds, ("lower", "upper"), strict=True):
                    same(name, value, reported[f"{field}_{endpoint}"])
            if name == "naive_minus_default_at_0.8":
                print(f"{generator}, naive minus default at T=0.8: {mean(delta):+.3f} "
                      f"[{float(reported['ci95_lower']):+.3f}, {float(reported['ci95_upper']):+.3f}]")
    print("Studies C/D: verified 30 count contrasts, 30 pointwise intervals and 10 family-sensitivity intervals.")


def supplementary_measures():
    supplement = json.loads((DATA / "main_local_metrics_posthoc_20261115.json").read_text())
    metrics = ("partition_count_proxy", "functional_classes", "reference_novelty",
               "novel_share_proxy", "functional_novel_share")
    print("Study A supplementary means:", ", ".join(metrics))
    cells = supplement["cells"]
    for arm in sorted({cell["arm"] for cell in cells}):
        selected = [cell for cell in cells if cell["arm"] == arm]
        repetitions = {(cell["task"], cell["repetition"]) for cell in selected}
        tasks = {task for task, _ in repetitions}
        if len(tasks) != 10 or len(selected) != 20 or repetitions != {(t, r) for t in tasks for r in (0, 1)}:
            raise ValueError(f"{arm}: incomplete task/repetition roster")
        print(arm, *(f"{mean(row[metric] for row in selected):.3f}" for metric in metrics))
    screen = json.loads((DATA / "small_generator_screen_20261117.json").read_text())
    for candidate in screen["candidates"]:
        coverage = candidate["coverage"]
        print(f"Study F {candidate['label']}: {coverage['generated_outputs']}/200 outputs, "
              f"{coverage['complete_response_sets']}/20 complete response sets")


if __name__ == "__main__":
    grouping_sensitivity()
    factorial_contrasts()
    supplementary_measures()
