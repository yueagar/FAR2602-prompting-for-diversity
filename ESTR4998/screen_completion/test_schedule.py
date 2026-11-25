from decimal import Decimal
import unittest

import schedule


def mock_job(identity, role, dependencies=()):
    return {"id": identity, "metadata": {"role": role}, "response_ids": list(dependencies)}


def mock_attempt(status, cost, **extra):
    return {"status": status, "reported_cost_usd": cost, "reserve_usd": .03, **extra}


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"jobs": [mock_job("mock-a", "generation"), mock_job("mock-b", "generation"),
                              mock_job("mock-q", "quality", ["mock-a"]),
                              mock_job("mock-g", "grouping", ["mock-a", "mock-b"])]}

    def test_exhausted_generation_blocks_only_dependent_judgments(self):
        failed = mock_attempt("failed", .01)
        records = [{"id": "mock-b", "attempts": [failed, failed]}]
        pending, missing = schedule.remaining_jobs(self.plan, records)
        self.assertEqual([row["id"] for row in pending], ["mock-a", "mock-q"])
        self.assertEqual(missing, ["mock-b", "mock-g"])

    def test_extended_retry_requires_a_specific_request(self):
        failed = mock_attempt("failed", .01)
        records = [{"id": "mock-b", "attempts": [failed, failed]}]
        self.plan["retry_amendment"] = {"generation_attempt_limits": {"mock-b": 3}}
        with self.assertRaisesRegex(ValueError, "single bound request"):
            schedule.remaining_jobs(self.plan, records, extended_generation_id="mock-a")
        pending, missing = schedule.remaining_jobs(self.plan, records, extended_generation_id="mock-b")
        self.assertEqual(missing, [])
        self.assertIn({"id": "mock-b", "maximum_additional_attempts": 1}, pending)

    def test_observed_reply_or_unresolved_charge_prevents_retry(self):
        for attempt in [mock_attempt("failed", .01, text=""), mock_attempt("failed", None)]:
            with self.subTest(attempt=attempt):
                _, missing = schedule.remaining_jobs(self.plan, [{"id": "mock-a", "attempts": [attempt]}])
                self.assertEqual(missing, ["mock-a", "mock-q", "mock-g"])

    def test_known_charges_and_unresolved_reserves(self):
        records = [{"attempts": [mock_attempt("ok", .02), mock_attempt("failed", None),
                                 mock_attempt("failed", .01)]}]
        account = schedule.accounting(records)
        self.assertEqual(account["attempted_requests"], 3)
        self.assertEqual(account["unknown_cost_attempts"], 1)
        self.assertAlmostEqual(account["known_cost_usd"], .03)
        self.assertAlmostEqual(account["committed_usd"], .06)
        self.assertEqual(schedule.failed_cost_usd(records), Decimal(".04"))
