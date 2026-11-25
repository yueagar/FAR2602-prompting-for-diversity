import copy
from datetime import datetime, timezone
import unittest

import completion


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.parent = [{"id": "mock-display-1", "role": "generation",
                        "attempts": [{"status": "failed", "reported_cost_usd": .01}]}]
        self.finished = copy.deepcopy(self.parent)
        self.finished[0]["attempts"].append({"status": "ok", "reported_cost_usd": .02})
        self.finished.append({"id": "mock-group", "role": "grouping",
                              "attempts": [{"status": "ok", "reported_cost_usd": .03}]})
        self.targets = {"mock-display-1": 1, "mock-group": 2}

    def tail(self):
        return completion.additional_attempts(self.parent, self.finished, self.targets,
                                              inherited_retry_id="mock-display-1")

    def test_reports_only_new_charges(self):
        tail = self.tail()
        self.assertEqual([len(row["attempts"]) for row in tail], [1, 1])
        self.assertAlmostEqual(sum(a["reported_cost_usd"] for row in tail for a in row["attempts"]), .05)
        self.assertEqual(len(self.parent[0]["attempts"]), 1)

    def test_missing_or_changed_history_is_rejected(self):
        self.finished[0]["attempts"][0]["reported_cost_usd"] = 1
        with self.assertRaisesRegex(ValueError, "prefix changed"):
            self.tail()
        self.finished.pop(0)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.tail()

    def test_inherited_retries_and_new_jobs_are_bounded(self):
        self.finished[0]["attempts"].append({"status": "ok"})
        with self.assertRaisesRegex(ValueError, "retry allowance"):
            self.tail()
        self.finished[0]["attempts"].pop()
        del self.targets["mock-group"]
        with self.assertRaisesRegex(ValueError, "outside completion"):
            self.tail()

    def test_transport_hints_keep_long_delays_and_parse_dates(self):
        clock = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
        self.assertEqual(completion.retry_after_seconds("00120", clock), 120)
        self.assertEqual(completion.retry_after_seconds("Wed, 01 Jan 2025 00:02:00 GMT", clock), 120)
        self.assertIsNone(completion.retry_after_seconds("malformed", clock))
        self.assertEqual(completion.retry_after_seconds("9" * 14, clock), {"unbounded_delay_seconds": "9" * 14})
