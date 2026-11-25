"""Verify submission files and replay selected results in temporary directories."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def check_inventory():
    entries = (HERE / "SHA256SUMS").read_text().splitlines()
    for entry in entries:
        digest, filename = entry.split("  ", 1)
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch for {filename}")
    print(f"Verified {len(entries)} release files.", flush=True)


def run(script, *args, env=None):
    subprocess.run([sys.executable, "-B", str(script), *map(str, args)], check=True, env=env)


def compare(files, expected, actual):
    for filename in files:
        if (expected / filename).read_bytes() != (actual / filename).read_bytes():
            raise ValueError(f"Reproduced file differs from the reference: {filename}")
    print(f"Reproduced {', '.join(files)}.", flush=True)


def full_replay(work):
    supplements = HERE / "local_model_study" / "supplements"
    unified = supplements / "20261117_unified_local"
    intervals = supplements / "20261117_interval_sensitivity"
    figures = HERE / "interim_report" / "figures"
    env = dict(os.environ, MPLCONFIGDIR=str(work / "mpl"), MPLBACKEND="Agg")
    run(unified / "analyze.py", "--output-dir", work / "unified")
    compare(("arm_means.csv", "task_metrics.csv", "contrasts.csv"), unified, work / "unified")
    run(intervals / "analyze.py", "--output-dir", work / "interval")
    compare(("summary.json", "contrasts.csv", "manifest.json", "RESULTS.md"), intervals, work / "interval")
    run(intervals / "plot.py", "--input", work / "interval" / "summary.json",
        "--output-dir", work / "figures", env=env)
    compare(("local_unified_sensitivity.png", "local_unified_sensitivity.svg"), figures, work / "figures")

    # The original plotting scripts use paths relative to themselves. Start each
    # output directory empty so a stale reference figure cannot pass this check.
    report = work / "interim_report"
    data = report / "data"
    charts = report / "figures"
    data.mkdir(parents=True)
    charts.mkdir()
    source_data = HERE / "interim_report" / "data"
    for filename in ("temperature_instruction_20261116.json", "second_generator_20261116.json"):
        shutil.copyfile(source_data / filename, data / filename)
    for script in ("generate_factorial_summary.py", "generate_cross_generator_summary.py", "generate_task_effects.py"):
        shutil.copyfile(figures / script, charts / script)
        run(charts / script, env=env)
    compare(tuple(f"{study}_{table}.csv" for study in ("temperature_instruction", "cross_generator")
                  for table in ("conditions", "contrasts", "task_counts")), source_data, data)
    compare(("temperature_instruction_summary.png", "temperature_instruction_summary.svg",
             "cross_generator_summary.pdf", "cross_generator_summary.png", "task_effects.pdf"), figures, charts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also rebuild the local-study tables and figures with the pinned packages")
    args = parser.parse_args()
    check_inventory()
    run(HERE / "evidence" / "study_a" / "reproduce.py")
    run(HERE / "interim_report" / "replay_results.py")
    opening = HERE / "interim_report" / "supplements" / "opening_reuse"
    table = subprocess.check_output([sys.executable, "-B", str(opening / "analyze.py")])
    if table != (opening / "RESULTS.md").read_bytes():
        raise ValueError("Opening-reuse table differs from its archived version")
    print("Reproduced the opening-reuse table.", flush=True)
    if args.full:
        with tempfile.TemporaryDirectory() as scratch:
            full_replay(Path(scratch))


if __name__ == "__main__":
    main()
