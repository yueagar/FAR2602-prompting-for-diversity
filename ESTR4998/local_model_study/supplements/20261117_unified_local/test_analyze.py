"""Synthetic tests of support, paired task estimands and input preservation."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("unified_local_analysis", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    cells = []
    for arm_index, arm in enumerate(analysis.ARMS):
        for task in range(10):
            for rep in range(3):
                metrics = {name: 0.5 for name in analysis.REQUIRED_METRICS}
                # Strong task variation cancels in the paired contrast. Repetition
                # variation must be averaged, never counted as extra tasks.
                metrics["functional_classes"] = task + rep / 10 + (task / 10 if arm == "naive_creative" else 0)
                metrics["vendi_score"] = task + rep / 10 + (1 if arm == "naive_creative" else 0)
                metrics["constraint_satisfaction"] = (task / 10 + arm_index / 100) if task < 6 else None
                metrics["quality_mean"] = 4.0
                metrics["quality_relevance"] = 4.0
                metrics["expected_quality"] = 3.8
                metrics["quality_parsed_share"] = 0.9
                metrics["expected_parsed_share"] = 0.8
                cells.append({"condition": arm, "prompt_id": f"task-{task:02d}", "replicate": rep, "metrics": metrics})
    return {"cells": cells}


class AnalysisTests(unittest.TestCase):
    def test_paired_task_estimand_and_intervals(self):
        summary = analysis.summarize(fixture())
        row = next(r for r in summary["contrasts"] if r["metric"] == "functional_classes" and r["condition"] == "naive_creative" and r["comparator"] == "default")
        self.assertAlmostEqual(row["difference"], 0.45)
        tasks = np.arange(10, dtype=float) / 10
        draws = np.random.default_rng(20261117).integers(0, 10, size=(20000, 10))
        expected = np.quantile(tasks[draws].mean(axis=1), [0.025, 0.975], method="linear")
        np.testing.assert_allclose([row["ci_low"], row["ci_high"]], expected)
        self.assertEqual(row["n_tasks"], 10)
        self.assertEqual(row["n_cells_per_arm"], 30)
        constant = next(r for r in summary["contrasts"] if r["metric"] == "vendi_score" and r["condition"] == "naive_creative" and r["comparator"] == "default")
        np.testing.assert_allclose([constant["difference"], constant["ci_low"], constant["ci_high"]], [1, 1, 1])
        self.assertEqual(len(summary["contrasts"]), len(summary["metrics"]) * 11)

    def test_eleven_unique_pairs_with_correct_signed_direction(self):
        data = fixture()
        for row in data["cells"]:
            if row["condition"] == "temp_0.5":
                row["metrics"]["functional_classes"] -= 2
        summary = analysis.summarize(data)
        rows = [row for row in summary["contrasts"] if row["metric"] == "functional_classes"]
        pairs = {(row["condition"], row["comparator"]): row for row in rows}
        expected = {(arm, "default") for arm in analysis.ARMS[1:]} | {
            ("naive_creative", arm) for arm in analysis.ARMS[1:-1]
        }
        self.assertEqual(len(rows), 11)
        self.assertEqual(set(pairs), expected)
        self.assertAlmostEqual(pairs[("temp_0.5", "default")]["difference"], -2)
        self.assertAlmostEqual(pairs[("naive_creative", "temp_0.5")]["difference"], 2.45)
        self.assertEqual(sum(row["comparison_role"] == "primary" for row in rows), 1)
        self.assertEqual(pairs[("naive_creative", "default")]["comparison_role"], "primary")

    def test_entire_missing_temperature_arm_is_rejected(self):
        data = fixture()
        data["cells"] = [row for row in data["cells"] if row["condition"] != "temp_0.5"]
        with self.assertRaises(ValueError):
            analysis.validate(data)

    def test_constraints_use_six_tasks_and_preserve_parse_shares(self):
        summary = analysis.summarize(fixture())
        constraint = next(r for r in summary["arm_means"] if r["condition"] == "default" and r["metric"] == "constraint_satisfaction")
        self.assertEqual(constraint["n_tasks"], 6)
        self.assertAlmostEqual(constraint["mean"], 0.25)
        missing = next(r for r in summary["task_metrics"] if r["condition"] == "default" and r["prompt_id"] == "task-09")
        self.assertIsNone(missing["constraint_satisfaction"])
        self.assertEqual(missing["quality_mean"], 4.0)
        self.assertAlmostEqual(missing["quality_parsed_share"], 0.9)

    def test_reject_duplicate_missing_and_wrong_replicates(self):
        data = fixture()
        for changed in (data["cells"] + [copy.deepcopy(data["cells"][0])], data["cells"][:-1]):
            with self.subTest(kind=len(changed)), self.assertRaises(ValueError):
                analysis.validate({"cells": changed})
        data["cells"][0]["replicate"] = True
        with self.assertRaises(ValueError):
            analysis.validate(data)

    def test_reject_missing_quality_and_nonfinite_or_boolean_metrics(self):
        for value in (None, float("nan"), float("inf"), True):
            data = fixture()
            data["cells"][0]["metrics"]["quality_mean"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                analysis.validate(data)

    def test_reject_inconsistent_metric_keys_and_constraint_support(self):
        data = fixture()
        del data["cells"][0]["metrics"]["expected_quality"]
        with self.assertRaises(ValueError):
            analysis.validate(data)
        data = fixture()
        data["cells"][0]["metrics"]["constraint_satisfaction"] = None
        with self.assertRaises(ValueError):
            analysis.validate(data)
        data = fixture()
        for row in data["cells"]:
            if row["prompt_id"] == "task-05":
                row["metrics"]["constraint_satisfaction"] = None
        with self.assertRaises(ValueError):
            analysis.validate(data)

    def test_declared_optional_diagnostics_only(self):
        data = fixture()
        for row in data["cells"]:
            row["metrics"].update({name: 0.5 for name in analysis.OPTIONAL_METRICS})
        _, names, _ = analysis.validate(data)
        self.assertTrue(analysis.OPTIONAL_METRICS.issubset(names))
        for row in data["cells"]:
            row["metrics"]["quality_undeclared"] = 0.5
        with self.assertRaises(ValueError):
            analysis.validate(data)

    def test_cli_help_exits_without_opening_inputs(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("analyze.py")), "--help"],
                                check=True, capture_output=True, text=True)
        self.assertIn("--input", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_outputs_reproduce_and_input_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cells.json"
            source.write_text(json.dumps(fixture()))
            original = source.read_bytes()
            analysis.write_outputs(source, root / "output")
            first = {name: (root / "output" / name).read_bytes() for name in analysis.OUTPUTS}
            analysis.write_outputs(source, root / "output")
            self.assertEqual(first, {name: (root / "output" / name).read_bytes() for name in analysis.OUTPUTS})
            self.assertEqual(source.read_bytes(), original)
            summary = json.loads(first["summary.json"])
            self.assertEqual(summary["grid"]["cells"], 210)
            self.assertIn("naive − default", first["RESULTS.md"].decode())

    def test_input_alias_and_output_alias_are_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cells.json"
            source.write_text(json.dumps(fixture()))
            original = source.read_bytes()
            os.link(source, root / "summary.json")
            with self.assertRaises(ValueError):
                analysis.write_outputs(source, root)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse((root / "arm_means.csv").exists())
            (root / "summary.json").unlink()
            (root / "summary.json").write_text("existing")
            (root / "arm_means.csv").symlink_to("summary.json")
            with self.assertRaises(ValueError):
                analysis.write_outputs(source, root)
            self.assertEqual((root / "summary.json").read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
