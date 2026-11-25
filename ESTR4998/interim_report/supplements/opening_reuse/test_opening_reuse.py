import csv
import itertools
import unittest

import analyze


class OpeningReuseTests(unittest.TestCase):
    def test_exact_duplicates_and_eight_token_cutoff(self):
        prefix = "One two THREE four five six seven eight"
        texts = [prefix + " then diverges", prefix.lower(), prefix] + ["other"] * 7
        self.assertEqual(analyze.set_counts(texts)["matching_pairs"], 3 + 21)
        self.assertEqual(analyze.set_counts(texts[::-1]), analyze.set_counts(texts))

    def test_short_text_and_tokenless_openings(self):
        texts = ["A.M.", "a m", "", "!!!", "文字", "don't", "don’t", "x-y", "x y", "２"]
        values = analyze.set_counts(texts)
        self.assertEqual(values["matching_pairs"], 1)
        self.assertEqual(values["planned_pairs"], 45)
        self.assertEqual(values["empty_openings"], 3)
        self.assertEqual(values["short_nonempty_openings"], 7)
        self.assertEqual(analyze.set_counts([""] * 10)["matching_pairs"], 0)
        self.assertEqual(analyze.set_counts(["a"] * 10)["matching_pairs"], 45)

    def test_missing_slot_is_different_from_a_returned_empty_string(self):
        with self.assertRaises(ValueError):
            analyze.set_counts([""] * 9)
        with self.assertRaises(ValueError):
            analyze.set_counts([""] * 9 + [None])

    def test_two_repetitions_have_no_cross_set_pairs(self):
        texts = [str(i) for i in range(10)]
        separate = analyze.set_counts(texts)
        sets = {("C", "arm", task, rep): separate for task in analyze.TASKS for rep in (0, 1)}
        self.assertEqual(2 * separate["planned_pairs"], 90)
        self.assertEqual(analyze.task_rates(sets, "C", "arm"), [0] * 10)
        pooled = list(itertools.combinations(texts + texts, 2))
        self.assertEqual(len(pooled), 190)
        self.assertEqual(sum(a == b for a, b in pooled), 10)

    def test_complete_roster_and_incompatible_pair_counts(self):
        with (analyze.HERE / "sets.csv").open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(analyze.checked_rows(rows)), 440)
        with self.assertRaises(ValueError):
            analyze.checked_rows(rows[:-1])
        with self.assertRaises(ValueError):
            analyze.checked_rows(rows[:-1] + [rows[0]])
        with self.assertRaises(ValueError):
            analyze.checked_rows(rows[1:] + [dict(rows[0], empty_openings=10, matching_pairs=1)])


if __name__ == "__main__":
    unittest.main()
