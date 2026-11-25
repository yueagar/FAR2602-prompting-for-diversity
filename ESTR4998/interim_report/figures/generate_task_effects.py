"""Draw the T=0.8 task figure from the two frozen factorial study CSVs."""
import csv
from math import isclose
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TASKS = {
    "curated-0": "Girl and dog", "curated-2": "Kansas romcom",
    "curated-6": "Lemur fable", "curated-7": "Fictional biography",
    "curated-8": "Underground monster", "curated-10": "Three-character story",
    "curated-16": "Magical forest", "curated-18": "Time traveller",
    "narrative-dev-01": "Lighthouse warning", "narrative-dev-02": "Memory currency",
}


def main():
    with (DATA / "cross_generator_task_counts.csv").open() as source:
        cells = {(r["generator"], r["task"], r["condition"]): float(r["primary_count"])
                 for r in csv.DictReader(source)}
    with (DATA / "cross_generator_contrasts.csv").open() as source:
        estimates = {r["generator"]: r for r in csv.DictReader(source)
                     if r["contrast"] == "naive_minus_default_at_0.8" and r["measure"] == "primary_count"}
    fig, ax = plt.subplots(figsize=(5.5, 4.25), layout="constrained")
    ax.axvline(0, c="#585858", linewidth=0.8)
    ax.axhline(9.6, c="#999999", linewidth=0.65)
    for generator, color, marker, offset in [
        ("DeepSeek", "#0072B2", "o", -0.20), ("Nemotron", "#A44410", "^", 0.20)
    ]:
        effects = [cells[generator, task, "naive_creative-t0p8"] - cells[generator, task, "default-t0p8"]
                   for task in TASKS]
        est = estimates[generator]
        if not isclose(mean(effects), float(est["mean"])) or int(est["tasks"]) != len(TASKS):
            raise ValueError(f"Task effects disagree with frozen contrast for {generator}")
        ys = [i + offset for i in range(len(TASKS))]
        ax.scatter(effects, ys, s=28, c=color, marker=marker, label=generator, zorder=3)
        summary_y = len(TASKS) + 0.3 + offset
        ax.hlines(summary_y, float(est["ci95_lower"]), float(est["ci95_upper"]), colors=color, lw=1.8)
        ax.scatter(mean(effects), summary_y, s=32, c=color, marker=marker, zorder=3)
    ax.set_yticks([*range(len(TASKS)), len(TASKS) + 0.3], [*TASKS.values(), "Mean (95% interval)"])
    ax.set(ylim=(10.95, -0.7), xlim=(-4.6, 8.7), xticks=[-4, -2, 0, 2, 4, 6, 8])
    ax.set_xlabel("Difference in judged groups per ten responses", fontsize=9)
    ax.tick_params(axis="both", labelsize=8.5, length=0, pad=4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#dedede", lw=0.5)
    ax.set_axisbelow(True)
    fig.suptitle("Naive instruction minus default at temperature 0.8", fontsize=11, weight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=8.5)
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(HERE / "task_effects.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
