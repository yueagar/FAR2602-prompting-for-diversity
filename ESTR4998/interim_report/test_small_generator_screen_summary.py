"""Offline edge cases for the screen's fixed-denominator logical bounds."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from summarize_small_generator_screen import (
    count_vector_bounds, load_verified_report, threshold_status,
    verify_referenced_files,
)


class CountBoundsTests(unittest.TestCase):
    def test_missing_counts_cannot_remove_observed_ceiling_violation(self):
        result = count_vector_bounds([10, 10, 10, None, None, 1, 2, 3, 4, 5])
        self.assertEqual(result["planned_tasks"], 10)
        self.assertEqual(result["ceiling_share_lower_bound"], 0.3)
        self.assertEqual(result["ceiling_share_upper_bound"], 0.5)
        self.assertTrue(result["ceiling_rule_impossible"])

    def test_exact_threshold_with_missing_is_not_a_proven_violation(self):
        result = count_vector_bounds([10, 10, None, 1, 1, 1, 1, 1, 1, 1])
        self.assertFalse(result["ceiling_rule_impossible"])
        self.assertEqual(result["ceiling_share_lower_bound"], 0.2)
        self.assertEqual(result["ceiling_share_upper_bound"], 0.3)

    def test_all_missing_has_no_observed_failure_or_point_estimate(self):
        counts = [None] * 10
        result = count_vector_bounds(counts)
        self.assertEqual(result["observed_tasks"], 0)
        self.assertEqual(result["median_lower_bound"], 1)
        self.assertEqual(result["median_upper_bound"], 10)
        self.assertFalse(result["ceiling_rule_impossible"])
        self.assertEqual(counts, [None] * 10)

    def test_even_length_median_bounds_and_complete_vector(self):
        result = count_vector_bounds([1, 1, 2, 3, 4, 6, 7, 8, 9, 10])
        self.assertEqual(result["median_lower_bound"], 5)
        self.assertEqual(result["median_upper_bound"], 5)
        self.assertFalse(result["median_le_five_impossible"])

    def test_dropping_missing_task_is_rejected(self):
        with self.assertRaises(ValueError):
            count_vector_bounds([10] * 9)

    def test_invalid_counts_are_rejected(self):
        for invalid in (0, 11, 1.5, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                count_vector_bounds([invalid] + [None] * 9)


class ThresholdTests(unittest.TestCase):
    def test_known_successes_can_establish_floor_without_complete_data(self):
        self.assertEqual(threshold_status(188, 2, 180), "met_by_known_observations")

    def test_missing_repeat_sets_are_not_failures(self):
        self.assertEqual(threshold_status(15, 3, 18),
                         "unresolved_due_to_missing_observations")

    def test_observed_constraint_failure_cannot_be_reversed(self):
        self.assertEqual(threshold_status(99, 0, 108),
                         "unattainable_even_if_all_missing_succeed")


class ProvenanceTests(unittest.TestCase):
    def test_report_hash_required_and_modified_report_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            original = b'{"complete": false}\n'
            path.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            self.assertEqual(load_verified_report(path, digest), {"complete": False})
            path.write_text('{"complete": true}\n')
            with self.assertRaises(ValueError):
                load_verified_report(path, digest)

    def test_source_file_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / "source.json"
            path.write_text("original\n")
            report = {"files_sha256": {"source.json": hashlib.sha256(path.read_bytes()).hexdigest()}}
            self.assertEqual(verify_referenced_files(report, base), 1)
            path.write_text("changed\n")
            with self.assertRaises(ValueError):
                verify_referenced_files(report, base)

    def test_source_file_cannot_escape_run(self):
        with tempfile.TemporaryDirectory() as directory:
            report = {"files_sha256": {"../outside.json": "irrelevant"}}
            with self.assertRaises(ValueError):
                verify_referenced_files(report, Path(directory))


if __name__ == "__main__":
    unittest.main()
