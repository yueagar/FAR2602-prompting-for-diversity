"""Construct original request bodies against fictional tasks, answers and routes."""
import unittest

import adaptive_demo as engine
import request_plan as adapter


class RequestPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = engine.synthetic_plan()
        self.plan["context"]["live"] = {
            "routes": {
                "generator": {"model": "fixture/generator", "provider": "fixture"},
                "primary": {"model": "fixture/evaluator", "provider": "fixture",
                            "json_schema": True, "parameters": {}},
            },
            "generation": {"temperature": .7},
            "evaluation_profile": adapter.ev.TASK_SENSITIVE_PROFILE,
            "max_response_serialized_bytes": 1024,
            "max_instruction_serialized_bytes": 512,
        }

    def cell(self, instruction=None, phase="selection"):
        return engine.evaluation_request(self.plan, instruction or self.plan["seed"], 0, phase)

    def test_each_proposal_round_constructs_one_request(self):
        proposal = engine.proposal_request(self.plan, 1, [], [])
        jobs = adapter.expected_jobs(self.plan, proposal, {})
        self.assertEqual(len(jobs), 1)
        route, body, metadata, parser = next(iter(jobs.values()))
        self.assertEqual(metadata["demo_role"], "proposal")
        self.assertIn("two reusable", body["messages"][0]["content"])
        self.assertEqual(route["model"], "fixture/generator")
        self.assertIsNone(parser)

    def test_seed_matched_across_instructions_and_fresh_display_separate(self):
        def seeds(req):
            return [body["seed"] for _, body, _, _ in adapter.expected_jobs(self.plan, req, {}).values()]

        self.assertEqual(seeds(self.cell()), seeds(self.cell(self.plan["reference"])))
        self.assertEqual(len(set(seeds(self.cell()))), 5)
        self.assertTrue(set(seeds(self.cell())).isdisjoint(seeds(self.cell(phase="fresh_display"))))

    def test_six_judge_requests_require_five_valid_fabricated_answers(self):
        req = self.cell()
        generators = adapter.expected_jobs(self.plan, req, {})
        records = {key: {"attempts": [{"status": "ok", "reported_cost_usd": 0,
                                       "finish_reason": "stop", "reasoning_tokens": 0,
                                       "text": f"Fabricated sample {idx}."}]}
                   for idx, key in enumerate(generators)}
        planned = adapter.expected_jobs(self.plan, req, records)
        judges = [value for key, value in planned.items() if key not in generators]
        self.assertEqual(len(judges), 6)
        for _, body, role, parse in judges:
            self.assertNotIn(self.plan["seed"]["instruction"], body["messages"][-1]["content"])
            if role["role"] == "grouping":
                self.assertEqual(parse('{"groups":[[0,1,2,3,4]]}'), [0]*5)
            else:
                self.assertEqual(parse('{"score":7}'), 7)
        records[next(iter(generators))]["attempts"][0]["finish_reason"] = "length"
        self.assertEqual(list(adapter.expected_jobs(self.plan, req, records)), list(generators))
