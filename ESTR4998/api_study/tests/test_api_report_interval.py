"""Deterministic API-report bootstrap contracts, not a coverage simulation."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api_report as report


class StudyReportIntervalTest(unittest.TestCase):
    def test_fewer_than_two_tasks_do_not_produce_precision(self):
        for values in ([], [2]):
            with self.subTest(values=values):
                self.assertIsNone(report.interval(values, 'fixed-seed'))

    def test_constant_differences_have_exact_degenerate_interval(self):
        self.assertEqual(report.interval([2.5] * 4, 'constant'), [2.5, 2.5])
        self.assertEqual(report.interval([-2.5] * 4, 'constant'), [-2.5, -2.5])

    def test_two_task_discrete_bootstrap_is_seeded_and_leaves_input_rng_unchanged(self):
        values = [-2, 6]
        state = random.getstate()
        first = report.interval(values, 'two-task-fixture')
        self.assertEqual(first, [-2, 6])
        self.assertEqual(report.interval(values, 'two-task-fixture'), first)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(values, [-2, 6])
        # The n=2 bootstrap has only means -2, 2, 6; the seeded 95% interval
        # spans its extremes. This checks computation, not frequentist coverage.

    def test_real_intervals_keep_task_pairing_sign_and_row_order(self):
        def row(task, arm, repetition, count):
            return {'cell': {'generator': 'g', 'task': task, 'arm': arm, 'repetition': repetition},
                    'judges': {'primary': {'grouping_complete': True,
                                           'metrics': {'distinct_count': count}}}}
        rows = []
        for task, baseline, difference in [('a', (2, 4), 2), ('b', (1, 3), 1), ('c', (1, 2), 3)]:
            for repetition, count in enumerate(baseline):
                rows += [row(task, 'default', repetition, count),
                         row(task, 'candidate', repetition, count + difference)]

        def compare(records, reference='default', tasks=('a', 'b', 'c')):
            return report.paired_comparisons(records, ['default', 'candidate'], 2,
                                             list(tasks), seed=19, reference=reference)[0]

        forward = compare(rows)
        self.assertEqual(forward['per_task_differences'], [2, 1, 3])
        self.assertEqual(forward['mean_difference'], 2)
        self.assertEqual(forward['paired_task_bootstrap_95'], [1, 3])
        self.assertEqual(compare(list(reversed(rows))), forward)
        permuted = compare(rows, tasks=('c', 'a', 'b'))
        self.assertEqual(permuted['per_task_differences'], [3, 2, 1])
        self.assertEqual(permuted['paired_task_bootstrap_95'], [1, 3])
        reverse = compare(rows, reference='candidate')
        self.assertEqual(reverse['per_task_differences'], [-2, -1, -3])
        self.assertEqual(reverse['mean_difference'], -2)
        self.assertEqual(reverse['paired_task_bootstrap_95'], [-3, -1])


if __name__ == '__main__':
    unittest.main()
