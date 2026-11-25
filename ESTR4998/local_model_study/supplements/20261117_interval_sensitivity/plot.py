#!/usr/bin/env python3
"""Compare two interval methods for saved local default contrasts.

This post hoc sensitivity figure plots saved summaries only. Both methods
use the same ten paired task means; repeated markers are the same estimate,
not independent observations. Original figures and analyses stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[2] / "interim_report" / "figures"
ORDER = (
    "temp_0.5", "naive_creative", "temp_1.0", "temp_1.3",
    "minp_0.1_t1.5", "untruncated_t1.0",
)
LABELS = {
    "temp_0.5": "Temperature 0.5",
    "naive_creative": "Naive instruction",
    "temp_1.0": "Temperature 1.0",
    "temp_1.3": "Temperature 1.3",
    "minp_0.1_t1.5": "Min-p 0.1, T = 1.5",
    "untruncated_t1.0": "Untruncated, T = 1.0",
}
PANELS = (
    ("functional_classes", "A. Functional classes", "Difference in classes\nper 10 responses"),
    ("semantic_diversity", "B. Mean cosine distance", "Difference in mean\npairwise distance"),
    ("quality_mean", "C. Automatic quality", "Difference in greedy\nscore (1–5 scale)"),
)
METHODS = (
    ("bootstrap", "Percentile task bootstrap", "#246da1", "o", "-", -0.11),
    ("t", "Paired t sensitivity", "#b56318", "s", "--", 0.11),
)


def selected_contrasts(summary: dict) -> dict[tuple[str, str], dict]:
    """Require complete, finite, paired ten-task support before plotting."""
    if summary.get("method", {}).get("confidence") != 0.95:
        raise ValueError("The figure requires nominal 95% intervals")
    metrics = {panel[0] for panel in PANELS}
    selected = {}
    for row in summary.get("contrasts", []):
        if row.get("comparator") != "default" or row.get("metric") not in metrics:
            continue
        identity = (row.get("condition"), row["metric"])
        if identity in selected or identity[0] not in ORDER:
            raise ValueError("Duplicate or unexpected default contrast")
        fields = ("difference", "bootstrap_ci_low", "bootstrap_ci_high", "t_ci_low", "t_ci_high")
        values = [row.get(field) for field in fields]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("Nonfinite or nonnumeric contrast")
        if row.get("n_tasks") != 10 or type(row.get("n_tasks")) is not int:
            raise ValueError("The figure requires ten paired task means")
        if values[1] > values[2] or values[3] > values[4]:
            raise ValueError("Reversed interval endpoints")
        if not values[3] <= values[0] <= values[4]:
            raise ValueError("Paired t interval must contain its mean")
        selected[identity] = row
    if set(selected) != {(arm, metric) for arm in ORDER for metric in metrics}:
        raise ValueError("Missing plotted contrast")
    return selected


def render(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    rows = selected_contrasts(json.loads(summary_path.read_text()))
    targets = tuple(output_dir / f"local_unified_sensitivity.{extension}" for extension in ("png", "svg"))
    for path in targets:
        if path.is_symlink() or (path.exists() and path.stat().st_nlink != 1):
            raise ValueError("Figure output must not alias another file")
        if path.resolve() == summary_path.resolve():
            raise ValueError("Figure output must not overwrite its source")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
        "svg.fonttype": "none", "svg.hashsalt": "estr-local-interval-sensitivity-20261117",
    })
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 3.2), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.40, top=0.77, wspace=0.27)
    for ax, (metric, title, xlabel) in zip(axes, PANELS):
        ax.axvline(0, color="#555555", linewidth=1, linestyle=":", zorder=1)
        for index, arm in enumerate(ORDER):
            row = rows[(arm, metric)]
            if index % 2 == 0:
                ax.axhspan(index - 0.46, index + 0.46, color="#f2f4f6", zorder=0)
            for method, _, color, marker, linestyle, offset in METHODS:
                low, high = row[f"{method}_ci_low"], row[f"{method}_ci_high"]
                position = index + offset
                # Draw endpoints independently: percentile intervals need not
                # contain their observed point estimate.
                ax.hlines(position, low, high, color=color, linestyle=linestyle,
                          linewidth=1, zorder=2)
                ax.vlines([low, high], position - 0.055, position + 0.055,
                          color=color, linewidth=0.8, zorder=2)
                ax.scatter(row["difference"], position, color=color, marker=marker,
                           s=13, edgecolors="white", linewidths=0.4, zorder=3)
        ax.set(title=title, xlabel=xlabel)
        ax.set_yticks(range(len(ORDER)), [LABELS[arm] for arm in ORDER])
        ax.set_ylim(len(ORDER) - 0.45, -0.55)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="x", alpha=0.16)
        ax.margins(x=0.12)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Unified local responses: interval sensitivity", fontsize=11, y=0.98)
    fig.text(0.5, 0.885,
             "Post hoc comparison · Qwen2.5-1.5B-Instruct · differences from default on 10 development tasks",
             ha="center", fontsize=7.5, color="#444444")
    legend = [Line2D([0], [0], color=color, marker=marker, linestyle=linestyle,
                     linewidth=1, markersize=4, label=label)
              for _, label, color, marker, linestyle, _ in METHODS]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.15),
               ncol=2, frameon=False)
    fig.text(0.5, 0.025,
             "Both markers show the same task-mean difference. Bars are nominal 95% intervals;\n"
             "neither method is a validation. Automatic endpoints are unvalidated;\n"
             "intervals are not simultaneous. Untruncated sampling retains the 512-token cap.",
             ha="center", va="bottom", fontsize=7.5, color="#444444")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(targets[0], dpi=360, metadata={"Software": "Matplotlib; ESTR interval sensitivity"})
    fig.savefig(targets[1], metadata={"Date": None, "Creator": "ESTR interval sensitivity"})
    # Matplotlib pads SVG path lines; normalize whitespace for clean diffs.
    svg = targets[1].read_text()
    targets[1].write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    plt.close(fig)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    for path in render(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
