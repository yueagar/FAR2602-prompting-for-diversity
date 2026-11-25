"""Export already frozen estimates and draw a descriptive two-generator figure.

Pass --report once to extract Nemotron's public aggregate snapshot. Subsequent
runs use the two committed snapshots. No raw texts, inference, new intervals,
pooled effects, or network calls. Ten tasks remain each study's resampling unit.
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
REPORT_SHA = "bdf81bf2e4e47b4b29cc4f02b5b2c2159eb59bde1c68529d9af60ffe88e32f3d"
PLAN_SHA = "631414c72feaf5e783ff3d62792e4a1cc864964125b3f5487485724d86db929b"
FIRST_REPORT_SHA = "2ef6113cc965e9f75866bd102334d00fc460dae01442e859e8422bfe2a6f1549"
SNAPSHOT = DATA / "second_generator_20261116.json"
LABELS = {"default": "Default", "naive_creative": "Naive",
          "manual_causal_shift_v1": "Causal shift"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path):
    if digest(path) != REPORT_SHA:
        raise ValueError("Expected the verified completed Nemotron report")
    report = json.loads(path.read_text())
    results, analysis = report["results"], report["analysis"]
    if report["plan_sha256"] != PLAN_SHA or not all(results[k] is True for k in
            ("complete", "observations_complete", "accounting_complete")):
        raise ValueError("Incomplete or unexpected source")
    if results["completed"] != {"generation": 1800, "grouping": 414, "quality": 2340}:
        raise ValueError("Unexpected observation support")
    if results["accounting"]["unknown_cost_attempts"] != 0 or not report["development_only"]:
        raise ValueError("Unresolved costs or changed interpretation")
    snapshot = {
        "schema": "estr-frozen-second-generator-public-summary-v1",
        "source_report": "ESTR4998/api_study/runs/"
                         "second-generator-factorial-20261116-v1/factorial-report-embeddings.json",
        "source_report_sha256": REPORT_SHA,
        "source_manifest_sha256": report["manifest_sha256"],
        "plan_sha256": PLAN_SHA,
        "export_script_sha256": digest(Path(__file__)),
        "scope": "Transcription of frozen estimates; cross-generator discussion is descriptive.",
        **{k: report[k] for k in ("development_only", "exploratory", "confirmatory",
                                  "main_eligible", "historical_failed_gates_preserved", "embedding_execution")},
        "completed": results["completed"], "accounting": results["accounting"],
        **{k: analysis[k] for k in ("analysis_protocol", "condition_summaries", "contrasts", "interpretation")},
    }
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return snapshot


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tables(studies, output):
    conditions, contrasts, tasks = [], [], []
    for model, study in studies.items():
        for s in study["condition_summaries"]:
            e, g = s["estimates"], s["generation_diagnostics"]
            conditions.append({"generator": model, "condition": s["condition"],
                "instruction": s["instruction_id"], "temperature": s["temperature"],
                "primary_count": e["primary_count"]["mean"], "quality": e["primary_quality"]["mean"],
                "constraint_passed": g["constraint_passed_outputs"],
                "constraint_applicable": g["constraint_applicable_outputs"],
                "ceiling_observation_0": s["ceiling_cells_by_judgment"][0],
                "ceiling_observation_1": s["ceiling_cells_by_judgment"][1],
                "sets_per_observation": s["planned_cells"], "same_partition": s["same_partition_cells"],
                "mean_words": g["mean_words_successful_outputs"]})
            for task, value in e["primary_count"]["per_task"].items():
                tasks.append({"generator": model, "condition": s["condition"], "task": task,
                    "primary_count": value, "quality": e["primary_quality"]["per_task"][task],
                    "constraint_rate": e["constraint_satisfied"]["per_task"].get(task, "")})
        for c in study["contrasts"]:
            for metric, estimate in c["estimates"].items():
                ci = estimate["paired_task_bootstrap_95"]
                family = estimate.get("approximate_family_bootstrap_99", ["", ""])
                guard_name = {"primary_quality": "quality", "constraint_satisfied": "constraint_satisfaction"}.get(metric)
                contrasts.append({"generator": model, "contrast": c["id"], "role": c["analysis_role"],
                    "measure": metric, "mean": estimate["mean"], "tasks": estimate["n_tasks"],
                    "ci95_lower": ci[0], "ci95_upper": ci[1],
                    "ci99_lower": family[0], "ci99_upper": family[1],
                    "automatic_safeguard": c["automatic_safeguards"].get(guard_name, {}).get("verdict", "not_applicable")})
    for name, rows in (("conditions", conditions), ("contrasts", contrasts), ("task_counts", tasks)):
        write_csv(output / "data" / f"cross_generator_{name}.csv", rows)


def plot(studies, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "estr-cross-generator-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12, 9.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=.55, wspace=.25)
    colors = {"default": "#334155", "naive_creative": "#0072b2", "manual_causal_shift_v1": "#d55e00"}
    markers = {"default": "o", "naive_creative": "s", "manual_causal_shift_v1": "^"}
    for col, (model, study) in enumerate(studies.items()):
        ax = fig.add_subplot(grid[0, col])
        for instruction, label in LABELS.items():
            rows = sorted((s for s in study["condition_summaries"] if s["instruction_id"] == instruction),
                          key=lambda s: s["temperature"])
            ax.plot([s["temperature"] for s in rows], [s["estimates"]["primary_count"]["mean"] for s in rows],
                    color=colors[instruction], marker=markers[instruction], label=label)
        ax.axhline(10, color="#777777", ls=":", lw=1)
        ax.set(title=f"{model}: descriptive condition means", xlabel="Requested temperature",
               ylabel="Mean Hydrogen groups / 10", xticks=[.8, 1, 1.2], ylim=(0, 10.5))
        ax.legend(frameon=False, loc="lower right")
    ax = fig.add_subplot(grid[1, :])
    names = ["Naive 0.8 − default 0.8", "Naive 1.0 − default 1.0", "Naive 1.2 − default 1.2",
             "Naive 0.8 − default 1.0", "Naive 0.8 − default 1.2"]
    model_colors = ["#0072b2", "#d55e00"]
    for index, (model, study) in enumerate(studies.items()):
        primary = [c for c in study["contrasts"] if c["analysis_role"] == "primary"]
        for row, c in enumerate(primary):
            e, y = c["estimates"]["primary_count"], row + (-.16 if index == 0 else .16)
            ax.hlines(y, *e["approximate_family_bootstrap_99"], color=model_colors[index], lw=1.6)
            ax.hlines(y, *e["paired_task_bootstrap_95"], color=model_colors[index], lw=5)
            ax.plot(e["mean"], y, marker="o" if index == 0 else "s", color=model_colors[index], ms=5)
    ax.axvline(0, color="#777777", ls="--", lw=1)
    ax.set(title="All five primary contrasts: separate within-generator estimates",
           xlabel="Naive − default: difference in mean judged groups", yticks=range(5), yticklabels=names,
           ylim=(4.7, -.7), xlim=(-1.5, 5.5))
    ax.legend(handles=[Line2D([0], [0], color=c, lw=3, marker=m, label=n)
                       for n, c, m in zip(studies, model_colors, ["o", "s"])], loc="lower right", frameon=False)
    for panel in fig.axes:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="x", alpha=.12)
    fig.subplots_adjust(left=.22, right=.97, top=.92, bottom=.13)
    fig.suptitle("Instruction effects across two generators: exploratory development results", fontsize=14)
    fig.text(.08, .04, "Bars: thick = nominal 95%; thin = approximate 99% primary-family sensitivity. Ten tasks per study.\n"
             "No pooled or cross-model effect test. Shared tasks and evaluators; automatic counts do not validate creativity.", fontsize=9)
    target = output / "figures" / "cross_generator_summary"
    fig.savefig(target.with_suffix(".png"), dpi=220)
    fig.savefig(target.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent)
    args = parser.parse_args()
    second = extract(args.report) if args.report else json.loads(SNAPSHOT.read_text())
    first = json.loads((DATA / "temperature_instruction_20261116.json").read_text())
    if first["source_report_sha256"] != FIRST_REPORT_SHA or second["source_report_sha256"] != REPORT_SHA:
        raise ValueError("Unexpected snapshot source identity")
    if first["analysis_protocol"] != second["analysis_protocol"]:
        raise ValueError("The two frozen analysis protocols differ")
    studies = {"DeepSeek": first, "Nemotron": second}
    for study in studies.values():
        if len(study["condition_summaries"]) != 9 or len(study["contrasts"]) != 15:
            raise ValueError("Missing frozen conditions or contrasts")
    for folder in ("data", "figures"):
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    tables(studies, args.output_dir)
    plot(studies, args.output_dir)
    print("Exported both studies' frozen estimates to three CSVs and PNG/PDF; no new inference.")


if __name__ == "__main__":
    main()
