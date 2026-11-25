"""Extract a fixed terminal screen report without model calls or source writes.

The original candidate summaries, including null complete-data quantities, are
retained. Derived bounds are post hoc logical checks over fixed denominators,
not imputed observations, inferential intervals, or replacement frozen gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path


TERMINAL_SHA256 = "5252063a0383e3b75250aa3ea573f6242e715be117044f8ba783488b1c171cc3"


def load_verified_report(path: Path, expected_sha256: str) -> dict:
    """Require the caller's expected byte hash before interpreting the report."""
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Terminal report SHA-256 mismatch: {actual}")
    return json.loads(payload)


def verify_referenced_files(report: dict, directory: Path) -> int:
    """Verify every source-record hash while rejecting paths outside the run."""
    base = directory.resolve()
    for name, expected in report["files_sha256"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"Source path escapes the run: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Referenced source SHA-256 mismatch: {name}")
    return len(report["files_sha256"])


def count_vector_bounds(counts: list[int | None], *, task_count: int = 10,
                        response_count: int = 10) -> dict:
    """Bound complete-vector summaries without changing missing observations.

    A missing partition could have between one and response_count groups.
    The observed number of ceilings is a lower bound over all planned tasks.
    Median extrema describe possible completions only; no completion is saved
    as data. Each vector retains its original task denominator.
    """
    if len(counts) != task_count or task_count < 1 or response_count < 1:
        raise ValueError("Count vector does not retain its planned denominator")
    if any(x is not None and (type(x) is not int or not 1 <= x <= response_count)
           for x in counts):
        raise ValueError("Counts must be integers in [1, response_count] or null")
    missing = counts.count(None)
    ceiling = sum(x == response_count for x in counts)
    low_median = statistics.median([1 if x is None else x for x in counts])
    high_median = statistics.median([
        response_count if x is None else x for x in counts
    ])
    return {
        "planned_tasks": task_count,
        "observed_tasks": task_count - missing,
        "missing_tasks": missing,
        "observed_ceiling_tasks": ceiling,
        "ceiling_share_lower_bound": ceiling / task_count,
        "ceiling_share_upper_bound": (ceiling + missing) / task_count,
        "median_lower_bound": low_median,
        "median_upper_bound": high_median,
        "ceiling_rule_impossible": ceiling / task_count > 0.20,
        "median_le_seven_impossible": low_median > 7,
        "median_le_five_impossible": low_median > 5,
    }


def threshold_status(known_successes: int, missing: int, threshold: int) -> str:
    """Distinguish an established threshold violation from missing support."""
    if any(type(x) is not int or x < 0
           for x in (known_successes, missing, threshold)):
        raise ValueError("Threshold counts must be non-negative integers")
    if known_successes >= threshold:
        return "met_by_known_observations"
    if known_successes + missing < threshold:
        return "unattainable_even_if_all_missing_succeed"
    return "unresolved_due_to_missing_observations"


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(report: dict, source_sha256: str, files_verified: int,
              source_path: str) -> dict:
    accounting = report["results"]
    analysis = report["screen_report"]["analysis"]
    if not accounting["schedule_exhausted"] or accounting["remaining_jobs"]:
        raise ValueError("Source is not a terminal exhausted-schedule report")
    if not accounting["accounting_complete"]:
        raise ValueError("Terminal source has unresolved accounting")
    if report["selected_candidate"] is not None or report["follow_on_authorized"]:
        raise ValueError("Unexpected selected candidate or follow-on authorization")
    if report["confirmatory"] or not report["exploratory"]:
        raise ValueError("Source is not the exploratory screen")
    candidates = []
    for original in analysis["candidates"]:
        cells = [x for x in analysis["cells"]
                 if x["cell"]["generator"] == original["label"]]
        if len(cells) != original["expected_sets"]:
            raise ValueError("Candidate does not retain every planned response set")
        pairs = [x["primary_repeat_agreement"] for x in cells
                 if x["primary_repeat_agreement"] is not None]
        cross = [v for x in cells for v in x["check_vs_primary"] if v is not None]
        quality_observed = sum(v is not None for x in cells
                               for v in x["primary_quality"])
        complete_sets = sum(x["generation_complete"] for x in cells)
        full_tasks = sum(all(x["generation_complete"] for x in cells
                            if x["cell"]["task"] == task)
                         for task in {x["cell"]["task"] for x in cells})
        bounds = [{"repetition": v["repetition"],
                   "observation": v["observation"],
                   **count_vector_bounds(v["counts"])}
                  for v in original["count_vectors"]]
        candidates.append({
            "label": original["label"],
            "source_candidate": copy.deepcopy(original),
            "coverage": {
                "generated_outputs": original["observed_outputs"],
                "quality_scores": quality_observed,
                "hydrogen_partitions": sum(v is not None for x in cells
                                            for v in x["primary_partitions"]),
                "luna_partitions": sum(x["check_partition"] is not None
                                        for x in cells),
                "complete_response_sets": complete_sets,
                "tasks_with_both_sets_complete": full_tasks,
            },
            "posthoc_logical_bounds": bounds,
            "threshold_support": {
                "quality_ge_180_of_200": threshold_status(
                    original["quality_at_least_six"],
                    original["expected_outputs"] - quality_observed, 180),
                "constraints_ge_108_of_120": threshold_status(
                    original["constraint_passes"],
                    original["constraint_expected"] - original["constraint_observed"],
                    108),
                "repeat_count_close_ge_18_of_20": threshold_status(
                    original["repeat_count_close_sets"],
                    original["expected_sets"] - len(pairs), 18),
            },
            "available_set_diagnostics": {
                "hydrogen_repeat_pairs": len(pairs),
                "hydrogen_exact_count_pairs": sum(
                    x["absolute_count_difference"] == 0 for x in pairs),
                "hydrogen_exact_partition_pairs": sum(
                    x["exact_partition_equal"] for x in pairs),
                "hydrogen_mean_ari": _mean([x["adjusted_rand_index"] for x in pairs]),
                "hydrogen_mean_absolute_count_difference": _mean([
                    x["absolute_count_difference"] for x in pairs]),
                "luna_hydrogen_comparisons": len(cross),
                "luna_hydrogen_exact_partition_pairs": sum(
                    x["exact_partition_equal"] for x in cross),
                "luna_hydrogen_mean_ari": _mean([
                    x["adjusted_rand_index"] for x in cross]),
                "luna_hydrogen_mean_absolute_count_difference": _mean([
                    x["absolute_count_difference"] for x in cross]),
                "complete_sets_with_exact_duplicates": sum(
                    x["exact_duplicate_rate"] > 0 for x in cells
                    if x["generation_complete"]),
            },
        })
    observed = {
        "generation": sum(x["coverage"]["generated_outputs"] for x in candidates),
        "quality": sum(x["coverage"]["quality_scores"] for x in candidates),
        "grouping": sum(x["coverage"]["hydrogen_partitions"]
                        + x["coverage"]["luna_partitions"] for x in candidates),
    }
    if observed != accounting["completed"]:
        raise ValueError("Candidate coverage disagrees with terminal accounting")
    return {
        "schema": "estr-small-generator-screen-terminal-summary-v1",
        "scope": "Preserved terminal snapshot; no imputation or new inferential test",
        "source": {
            "terminal_report_path": source_path,
            "terminal_report_sha256": source_sha256,
            "referenced_source_files_verified": files_verified,
            "binding_file_sha256": report["binding_sha256"],
            "scientific_plan_sha256": report["screen_report"]["plan_sha256"],
            "manifest_sha256": report["screen_report"]["manifest_sha256"],
        },
        "terminal_accounting": copy.deepcopy(accounting),
        "source_protocol": copy.deepcopy(analysis["protocol"]),
        "source_execution_readiness": copy.deepcopy(analysis["execution_readiness"]),
        "source_complete": analysis["complete"],
        "source_selected_candidate": analysis["selected_candidate"],
        "source_follow_on_authorized": analysis["follow_on_authorized"],
        "candidates": candidates,
        "interpretation": {
            "bound_status": "Post hoc logical checks; original null summaries retained",
            "missingness": "Missing observations are not measured failures",
            "judge_scope": "Automatic exploratory diagnostics, not human validation",
            "study_scope": "Three selected models, two families, ten reused development tasks",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.report.resolve().parent):
        parser.error("Write the summary outside the preserved source run")
    report = load_verified_report(args.report, args.expected_sha256)
    verified = verify_referenced_files(report, args.report.parent)
    repository = Path(__file__).resolve().parents[2]
    try:
        source_path = args.report.resolve().relative_to(repository).as_posix()
    except ValueError:
        parser.error("Keep the source report within the repository for relative provenance")
    output = summarize(report, args.expected_sha256, verified, source_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Verified {verified} source files; wrote {args.output}")


if __name__ == "__main__":
    main()
