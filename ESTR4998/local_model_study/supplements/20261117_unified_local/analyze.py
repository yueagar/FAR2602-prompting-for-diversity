#!/usr/bin/env python3
"""Analyze the frozen saved-output supplement without model or network access.

Each cell averages ten saved responses or measures their set. Average the three
cells within a task, then weight tasks equally. A contrast resamples *the same
task indices* for both arms; repetitions and individual responses are not
resampled. This deliberately differs from the historical two-level bootstrap.
Original records and reports are never rewritten by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ARMS = (
    "default", "temp_0.5", "temp_1.0", "temp_1.3", "minp_0.1_t1.5",
    "untruncated_t1.0", "naive_creative",
)
# Every nondefault arm against default, plus naive against the five other
# nondefault decoders. The focal naive/default pair appears exactly once.
CONTRASTS = tuple((arm, "default") for arm in ARMS[1:]) + tuple(
    ("naive_creative", arm) for arm in ARMS[1:-1]
)
REQUIRED_METRICS = (
    "functional_classes", "semantic_diversity", "reference_novelty",
    "distinct_1", "distinct_2", "novel_share_proxy", "vendi_score", "rke_score", "quality_mean",
    "quality_parsed_share", "expected_quality", "expected_parsed_share",
    "constraint_satisfaction", "completion_tokens_mean", "truncated_share",
    "mixed_script_share",
)
OPTIONAL_METRICS = frozenset({
    "partition_count_proxy", "functional_novel_share",
    "functional_window_truncated_share", "exact_duplicate_rate", "empty_share",
    "very_short_share", "repetitive_share", "instruction_leak_share",
    "refusal_share", "ai_disclaimer_share", "quality_relevance",
    "quality_coherence", "quality_constraints", "quality_effectiveness",
})
RESAMPLES = 20_000
SEED = 20261117
OUTPUTS = ("summary.json", "arm_means.csv", "contrasts.csv", "task_metrics.csv", "RESULTS.md")


def validate(payload: dict[str, Any]) -> tuple[list[str], list[str], dict[tuple[str, str, int], dict]]:
    """Require the complete seven-arm grid and consistent metric eligibility.

Null constraint values mean the task has no countable constraint. Every other
cell metric must be finite and available; missing quality is never imputed as
zero, or silently dropped to produce a comparison on different support.
"""
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("cells must be a nonempty list")
    indexed: dict[tuple[str, str, int], dict] = {}
    metrics: set[str] | None = None
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("each cell must be an object")
        arm, prompt, replicate = (cell.get(k) for k in ("condition", "prompt_id", "replicate"))
        if arm not in ARMS or not isinstance(prompt, str) or not prompt:
            raise ValueError("unknown arm or invalid prompt ID")
        if type(replicate) is not int or replicate not in (0, 1, 2):
            raise ValueError("replicate must be 0, 1 or 2")
        key = (arm, prompt, replicate)
        if key in indexed:
            raise ValueError(f"duplicate cell: {key}")
        values = cell.get("metrics")
        if not isinstance(values, dict) or not all(isinstance(k, str) for k in values):
            raise ValueError("metrics must be a string-keyed object")
        if metrics is None:
            metrics = set(values)
            missing = set(REQUIRED_METRICS) - metrics
            extras = metrics - set(REQUIRED_METRICS) - OPTIONAL_METRICS
            if missing or extras:
                raise ValueError(f"unexpected metric schema; missing={sorted(missing)}, extra={sorted(extras)}")
        elif set(values) != metrics:
            raise ValueError("metric keys differ across cells")
        for name, value in values.items():
            if value is None:
                if name != "constraint_satisfaction":
                    raise ValueError(f"missing nonconstraint metric {name} in {key}")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"nonfinite or nonnumeric metric {name} in {key}")
        indexed[key] = cell
    prompts = sorted({key[1] for key in indexed})
    expected = {(arm, prompt, rep) for arm in ARMS for prompt in prompts for rep in range(3)}
    if len(prompts) != 10 or set(indexed) != expected:
        raise ValueError("expected exactly seven arms × ten tasks × three repetitions")
    eligible = []
    for prompt in prompts:
        available = {
            indexed[(arm, prompt, rep)]["metrics"]["constraint_satisfaction"] is not None
            for arm in ARMS for rep in range(3)
        }
        if len(available) != 1:
            raise ValueError(f"constraint availability differs across arms or repetitions: {prompt}")
        if available == {True}:
            eligible.append(prompt)
    if len(eligible) != 6:
        raise ValueError("expected exactly six tasks with countable constraints")
    return prompts, sorted(metrics), indexed


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    prompts, metrics, indexed = validate(payload)
    task_rows = []
    for arm in ARMS:
        for prompt in prompts:
            values = {}
            for metric in metrics:
                observations = [indexed[(arm, prompt, rep)]["metrics"][metric] for rep in range(3)]
                values[metric] = None if observations[0] is None else float(np.mean(observations))
            task_rows.append({"condition": arm, "prompt_id": prompt, **values})
    task_map = {(row["condition"], row["prompt_id"]): row for row in task_rows}
    means, contrasts = [], []
    for metric in metrics:
        supported = [p for p in prompts if task_map[("default", p)][metric] is not None]
        matrix = np.array([[task_map[(arm, p)][metric] for p in supported] for arm in ARMS], dtype=float)
        # Common task draws retain the within-task pairing in every contrast.
        draws = np.random.default_rng(SEED).integers(0, len(supported), size=(RESAMPLES, len(supported)))
        resampled = matrix[:, draws].mean(axis=2)
        for index, arm in enumerate(ARMS):
            low, high = np.quantile(resampled[index], [0.025, 0.975], method="linear")
            means.append({"condition": arm, "metric": metric, "mean": float(matrix[index].mean()),
                          "ci_low": float(low), "ci_high": float(high),
                          "n_tasks": len(supported), "n_cells": len(supported) * 3})
        for condition, comparator in CONTRASTS:
            difference = matrix[ARMS.index(condition)] - matrix[ARMS.index(comparator)]
            sampled_difference = difference[draws].mean(axis=1)
            low, high = np.quantile(sampled_difference, [0.025, 0.975], method="linear")
            contrasts.append({
                "condition": condition, "comparator": comparator, "metric": metric,
                "comparison_role": "primary" if (condition, comparator) == ("naive_creative", "default") else "secondary",
                "metric_role": "historical_main_endpoint" if metric == "functional_classes" else "diagnostic",
                "difference": float(difference.mean()), "ci_low": float(low), "ci_high": float(high),
                "n_tasks": len(supported), "n_cells_per_arm": len(supported) * 3,
            })
    return {
        "schema": "estr-unified-local-v04-v05-v1",
        "method": {
            "scope": "retrospective unified v0.4/v0.5 saved-output development analysis",
            "focal_contrast": ["naive_creative", "default"],
            "declared_contrasts": [list(pair) for pair in CONTRASTS],
            "estimator": "equal repetition-cell means within tasks, then equal task means",
            "interval": "paired task percentile bootstrap; no within-task resampling",
            "resamples": RESAMPLES, "seed": SEED, "confidence": 0.95, "quantile_method": "linear",
            "multiplicity": "nominal intervals; no simultaneous coverage or p-values",
            "historical_intervals": "unchanged; historical two-level intervals use the same point estimator and another resampling procedure",
            "quality": "automatic scores with parse shares; no non-inferiority verdicts",
        },
        "grid": {"arms": list(ARMS), "tasks": prompts, "replicates": [0, 1, 2], "cells": len(indexed)},
        "metrics": metrics, "arm_means": means, "contrasts": contrasts, "task_metrics": task_rows,
    }


def csv_text(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Unified local v0.4/v0.5 results with common measurements", "",
        "This is a retrospective development-data supplement using saved responses. It is not a new generation experiment, a preregistered confirmation, or an independent replication. Original v0.4/v0.5 records and reports remain unchanged.", "",
        "The focal comparison is naive minus default, with classifier-defined functional classes as the historical main endpoint. Other measures are diagnostics. All six nondefault arms are compared with default; naive is also compared with each of the five other nondefault decoders. These are eleven unique contrasts. The ten nonfocal comparisons are secondary. They compare intervention packages; naive was not generated under the other decoders, so these do not estimate prompting × decoding interactions.", "",
        "The seven unique conditions contain 2,100 saved generation observations in 210 cells; the 900 shared v0.4/v0.5 observations are counted once. Each arm has ten tasks and three ten-response sets per task. Cell values are averaged within each task, then tasks receive equal weight. Constraint results use only the six eligible tasks; unconstrained tasks are missing, not passes. Parse shares remain separate from parsed quality means.", "",
        f"Intervals use {RESAMPLES:,} paired task bootstrap draws, seed {SEED}, and linear 2.5th/97.5th percentiles. Repetitions and responses are not resampled. These nominal 95% intervals differ from the historical two-level bootstrap and do not have simultaneous coverage across metrics or comparisons. Ten purposively chosen tasks do not justify population-wide generalization.", "",
        "## Arm means", "",
        "| Metric | " + " | ".join(ARMS) + " | Tasks |",
        "|---|" + "---:|" * (len(ARMS) + 1),
    ]
    lookup = {(row["condition"], row["metric"]): row for row in summary["arm_means"]}
    for metric in summary["metrics"]:
        values = [lookup[(arm, metric)] for arm in ARMS]
        lines.append("| " + metric + " | " + " | ".join(f"{x['mean']:.4f}" for x in values) + f" | {values[0]['n_tasks']} |")
    lines += ["", "Rates are proportions from zero to one. Quality scores use the local 1–5 scale; completion lengths are generated tokens. Full-precision means and intervals are in `arm_means.csv`.", ""]
    for condition, comparator in CONTRASTS:
        role = "Focal comparison" if (condition, comparator) == ("naive_creative", "default") else "Secondary comparison"
        label = "naive" if condition == "naive_creative" else condition
        lines += [f"## {role}: {label} − {comparator}", "", "| Metric | Difference [95% interval] | Tasks |", "|---|---:|---:|"]
        for row in summary["contrasts"]:
            if (row["condition"], row["comparator"]) == (condition, comparator):
                lines.append(f"| {row['metric']} | {row['difference']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['n_tasks']} |")
        lines.append("")
    lines += [
        "## Interpretation limits", "",
        "- Functional classes reflect the local classifier and its opening-only input window; they are not human-validated creative ideas.",
        "- Vendi, RKE and other embedding measures share a representation and can respond to length or superficial form. They are not independent confirmations, and effective counts are bounded by set size.",
        "- The local quality judge is unvalidated and can reward incoherent or noncompliant text. Probability-weighted scores add resolution without proving validity. No quality-preservation or non-inferiority verdict follows from these tables.",
        "- Deterministic constraints are limited heuristic checks. Length, truncation and mixed-script rates must be read alongside diversity; mixed-script flags are not a human-verified gibberish count.",
        "- Historical shared response texts are reused evidence. This supplement does not add independent observations or repair the original human-reliability failure.", "",
        "`summary.json` records the analysis method and full values; `task_metrics.csv` exposes every task mean; `contrasts.csv` contains every declared contrast. Input/source provenance and any local measurement recomputation are documented in the accompanying plan and input artifacts.", "",
    ]
    return "\n".join(lines)


def same_file(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (left.exists() and right.exists() and left.samefile(right))


def write_outputs(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate everything and detect source/output aliases before writing."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    targets = [output_dir / name for name in OUTPUTS]
    for index, target in enumerate(targets):
        if same_file(target, input_path) or any(same_file(target, other) for other in targets[:index]):
            raise ValueError("output must not alias an input or another output")
    original = input_path.read_bytes()
    summary = summarize(json.loads(original))
    summary["input"] = {"file": input_path.name, "sha256": hashlib.sha256(original).hexdigest()}
    rendered = {
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "arm_means.csv": csv_text(summary["arm_means"], list(summary["arm_means"][0])),
        "contrasts.csv": csv_text(summary["contrasts"], list(summary["contrasts"][0])),
        "task_metrics.csv": csv_text(summary["task_metrics"], ["condition", "prompt_id", *summary["metrics"]]),
        "RESULTS.md": markdown(summary),
    }
    if input_path.read_bytes() != original:
        raise ValueError("input changed during analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        target.write_text(rendered[target.name], encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("cells.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    summary = write_outputs(args.input, args.output_dir)
    print(json.dumps({"status": "complete", "cells": summary["grid"]["cells"], "metrics": len(summary["metrics"]),
                      "contrasts": len(summary["contrasts"]), "input_sha256": summary["input"]["sha256"]}))


if __name__ == "__main__":
    main()
