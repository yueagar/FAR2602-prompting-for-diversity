"""Export and plot the completed factorial's already-computed frozen estimates.

With --report, extract a public aggregate snapshot (no responses or judgments).
Without it, reproduce CSVs and figures from the committed snapshot alone.
No inference, new confidence intervals, endpoint changes or network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SNAPSHOT = DATA / "temperature_instruction_20261116.json"
PLAN_SHA = "9b2d70d4fce1b62612f0b59649e9a98a45d04c2be85845268cf709c78d437992"
LABELS = {
    "default": "Default",
    "naive_creative": "Naive",
    "manual_causal_shift_v1": "Causal shift",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_snapshot(path):
    report = json.loads(path.read_text())
    results, analysis = report["results"], report["analysis"]
    if report["plan_sha256"] != PLAN_SHA:
        raise ValueError("This export is specific to the approved November factorial")
    if not all(results[k] for k in ("complete", "observations_complete", "accounting_complete")):
        raise ValueError("Incomplete source report")
    if results["completed"] != {"generation": 1800, "grouping": 414, "quality": 2340}:
        raise ValueError("Unexpected completed-call counts")
    if results["accounting"]["unknown_cost_attempts"] != 0:
        raise ValueError("Unresolved accounting")
    if len(analysis["condition_summaries"]) != 9 or len(analysis["contrasts"]) != 15:
        raise ValueError("Unexpected analysis dimensions")
    if any(s["estimates"]["primary_count"]["n_tasks"] != 10
           for s in analysis["condition_summaries"]):
        raise ValueError("Incomplete primary task support")
    snapshot = {
        "schema": "estr-frozen-factorial-public-summary-v1",
        "source_report": "ESTR4998/api_study/runs/"
                         "factorial-20261115-v1/factorial-report-embeddings.json",
        "source_report_sha256": digest(path),
        "source_manifest_sha256": report["manifest_sha256"],
        "plan_sha256": report["plan_sha256"],
        "export_script_sha256": digest(Path(__file__)),
        "scope": "Transcription of frozen estimates; no new statistical analysis.",
        "development_only": report["development_only"],
        "confirmatory": report["confirmatory"],
        "historical_failed_gates_preserved": report["historical_failed_gates_preserved"],
        "embedding_execution": report["embedding_execution"],
        "completed": results["completed"],
        "accounting": results["accounting"],
        "analysis_protocol": analysis["analysis_protocol"],
        "condition_summaries": analysis["condition_summaries"],
        "contrasts": analysis["contrasts"],
        "interpretation": analysis["interpretation"],
    }
    DATA.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return snapshot


def write_csv(name, rows):
    with (DATA / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_tables(snapshot):
    conditions, tasks, contrasts = [], [], []
    for s in snapshot["condition_summaries"]:
        e, g = s["estimates"], s["generation_diagnostics"]
        conditions.append({
            "condition": s["condition"], "instruction": s["instruction_id"],
            "temperature": s["temperature"], "tasks": e["primary_count"]["n_tasks"],
            "primary_count": e["primary_count"]["mean"],
            "quality": e["primary_quality"]["mean"],
            "constraints_passed": g["constraint_passed_outputs"],
            "constraints_applicable": g["constraint_applicable_outputs"],
            "ceiling_observation_0": s["ceiling_cells_by_judgment"][0],
            "ceiling_observation_1": s["ceiling_cells_by_judgment"][1],
            "sets_per_observation": s["planned_cells"],
            "identical_repeated_partitions": s["same_partition_cells"],
            "mean_words": g["mean_words_successful_outputs"],
            **{k: e[k]["mean"] for k in ("partition_count_proxy", "vendi_score", "rke_score", "semantic_diversity")},
        })
        for task, count in e["primary_count"]["per_task"].items():
            tasks.append({"condition": s["condition"], "task": task, "primary_count": count,
                          "observation_0": e["first_judgment_count"]["per_task"][task],
                          "observation_1": e["second_judgment_count"]["per_task"][task]})
    for c in snapshot["contrasts"]:
        e, guards = c["estimates"], c["automatic_safeguards"]
        count = e["primary_count"]
        family = count.get("approximate_family_bootstrap_99", ["", ""])
        contrasts.append({
            "contrast": c["id"], "role": c["analysis_role"], "tasks": count["n_tasks"],
            "count_difference": count["mean"],
            "ci95_lower": count["paired_task_bootstrap_95"][0],
            "ci95_upper": count["paired_task_bootstrap_95"][1],
            "ci99_lower": family[0], "ci99_upper": family[1],
            "quality_difference": e["primary_quality"]["mean"],
            "quality_ci95_lower": e["primary_quality"]["paired_task_bootstrap_95"][0],
            "quality_ci95_upper": e["primary_quality"]["paired_task_bootstrap_95"][1],
            "quality_guard": guards.get("quality", {}).get("verdict", "not_applicable"),
            "constraint_difference": e["constraint_satisfied"]["mean"],
            "constraint_ci95_lower": e["constraint_satisfied"]["paired_task_bootstrap_95"][0],
            "constraint_ci95_upper": e["constraint_satisfied"]["paired_task_bootstrap_95"][1],
            "constraint_guard": guards.get("constraint_satisfaction", {}).get("verdict", "not_applicable"),
        })
    write_csv("temperature_instruction_conditions.csv", conditions)
    write_csv("temperature_instruction_task_counts.csv", tasks)
    write_csv("temperature_instruction_contrasts.csv", contrasts)


def plot(snapshot):
    # Use a task-specific writable cache; never download a model or font.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "estr-factorial-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
                         "svg.fonttype": "none", "svg.hashsalt": "estr-factorial-20261116"})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
    colors = {"default": "#334155", "naive_creative": "#0072b2", "manual_causal_shift_v1": "#d55e00"}
    markers = {"default": "o", "naive_creative": "s", "manual_causal_shift_v1": "^"}
    for instruction, label in LABELS.items():
        values = sorted((s for s in snapshot["condition_summaries"] if s["instruction_id"] == instruction),
                        key=lambda s: s["temperature"])
        axes[0].plot([s["temperature"] for s in values],
                     [s["estimates"]["primary_count"]["mean"] for s in values],
                     label=label, color=colors[instruction], marker=markers[instruction], linewidth=1.8)
    axes[0].axhline(10, color="#777777", linestyle=":", linewidth=1)
    axes[0].set(title="A. Condition means", xlabel="Generation temperature",
                ylabel="Mean Hydrogen groups\nper 10 responses", xticks=[0.8, 1.0, 1.2], ylim=(0, 10.5))
    axes[0].legend(loc="lower right", frameon=False)
    primary = [c for c in snapshot["contrasts"] if c["analysis_role"] == "primary"]
    names = ["Naive 0.8 − Default 0.8", "Naive 1.0 − Default 1.0", "Naive 1.2 − Default 1.2",
             "Naive 0.8 − Default 1.0", "Naive 0.8 − Default 1.2"]
    for y, c in enumerate(primary):
        e = c["estimates"]["primary_count"]
        axes[1].hlines(y, *e["approximate_family_bootstrap_99"], color="#a6bddb", linewidth=2)
        axes[1].hlines(y, *e["paired_task_bootstrap_95"], color="#0072b2", linewidth=5)
        axes[1].plot(e["mean"], y, "o", color="#172b4d", markersize=5)
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set(title="B. Five primary contrasts", xlabel="Difference in mean\njudged groups",
                yticks=range(5), yticklabels=names, ylim=(5.2, -0.6), xlim=(-0.4, 5.5))
    fig.legend(handles=[Line2D([0], [0], color="#0072b2", lw=5, label="95% pointwise"),
                        Line2D([0], [0], color="#a6bddb", lw=2, label="99% family sensitivity")],
               loc="lower center", bbox_to_anchor=(0.5, 0.155), ncol=2, frameon=False)
    for ax in axes:
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.81, bottom=0.40, wspace=0.99)
    fig.suptitle("Temperature × instruction: frozen exploratory results", fontsize=11)
    fig.text(0.5, 0.025,
             "Ten development tasks; two response sets and two grouping observations per set.\n"
             "Task-bootstrap intervals; 99% sensitivity is approximate.\n"
             "Automatic counts do not establish semantic validity.",
             ha="center", va="bottom", fontsize=7.5)
    fig.savefig(HERE / "temperature_instruction_summary.png", dpi=360)
    svg = HERE / "temperature_instruction_summary.svg"
    fig.savefig(svg, metadata={"Date": None})
    # Matplotlib emits trailing spaces in SVG paths; keep repository text clean.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Extract from this completed frozen report before rendering")
    args = parser.parse_args()
    snapshot = export_snapshot(args.report) if args.report else json.loads(SNAPSHOT.read_text())
    export_tables(snapshot)
    plot(snapshot)
    print("Exported three CSVs and PNG/SVG figures from the frozen factorial summary.")


if __name__ == "__main__":
    main()
