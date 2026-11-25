"""Behavioral tests of the adaptive demonstration; no API or credentials."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import adaptive_demo as demo


class AdaptiveDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = demo.synthetic_plan()

    def run_demo(self, *, plan=None, proposer=None, evaluator=None, name="run"):
        return demo.run_demo(plan or self.plan, self.root / name,
                             proposer or demo.synthetic_proposer, evaluator or demo.synthetic_evaluator,
                             evidence_mode="synthetic", adapter_id="test-v1")

    def test_adaptive_feedback_and_unfavorable_display_preserve_choice(self):
        history = []
        def proposer(request):
            history.append(copy.deepcopy(request))
            return demo.synthetic_proposer(request)
        result = self.run_demo(proposer=proposer)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["selection"]["selected_id"], "proposal-r2-s1")
        self.assertEqual(len(history), 2)
        self.assertNotIn("proposal-r1-s1", [r["id"] for r in history[0]["history"]])
        first = next(r for r in history[1]["history"] if r["id"] == "proposal-r1-s1")
        self.assertEqual(first["count"], 4)
        winner = next(c for c in result["selection"]["candidates"] if c["id"] == "proposal-r2-s1")
        display = next(c for c in result["fresh_display"]["summaries"] if c["id"] == "proposal-r2-s1")
        self.assertEqual(winner["count"], 5)
        self.assertEqual(winner["ceiling_cells"], 2)
        self.assertEqual(display["count"], 2)
        self.assertEqual(display["ceiling_cells"], 0)
        self.assertFalse(display["estimated_feasible"])
        self.assertFalse(result["effectiveness_claim"])
        self.assertEqual(result["selection"]["rejections"][0]["reason"], "duplicate_normalized_instruction")
        self.assertTrue((self.root / "run/trace.md").exists())
        self.assertIn("Ceiling cells /2", (self.root / "run/trace.md").read_text())

    def test_winner_frozen_before_first_display_callback(self):
        def evaluator(request):
            if request["phase"] == "fresh_display":
                frozen = demo.read(self.root / "run/selection.json")
                self.assertEqual(frozen["selected_id"], "proposal-r2-s1")
            return demo.synthetic_evaluator(request)
        self.run_demo(evaluator=evaluator)

    def test_resume_does_not_repeat_observed_callbacks(self):
        calls = []
        failed = False
        def evaluator(request):
            nonlocal failed
            calls.append(request["request_id"])
            if request["instruction"]["id"] == "proposal-r1-s1" and not failed:
                failed = True
                raise RuntimeError("simulated no-result interruption")
            return demo.synthetic_evaluator(request)
        with self.assertRaises(RuntimeError):
            self.run_demo(evaluator=evaluator)
        observed = {p.stem.removesuffix(".response") for p in (self.root / "run/calls").glob("*.response.json")}
        before = list(calls)
        report = self.run_demo(evaluator=evaluator)
        repeated = calls[len(before):]
        self.assertFalse(observed.intersection(repeated))
        self.assertEqual(report["status"], "completed")

    def test_observed_invalid_evaluation_stays_invalid_without_redraw(self):
        counts = 0
        def invalid(request):
            nonlocal counts
            counts += 1
            result = demo.synthetic_evaluator(request)
            result["samples"][0]["quality"] = float("nan")
            return result
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "Invalid quality"):
                self.run_demo(evaluator=invalid)
        self.assertEqual(counts, 1)
        responses = list((self.root / "run/calls").glob("*.response.json"))
        self.assertEqual(len(responses), 1)
        self.assertIn("NaN", responses[0].read_text())

    def test_missing_cell_stops_before_proposal_or_selection(self):
        proposal_calls = []
        def evaluator(request):
            result = demo.synthetic_evaluator(request)
            if request["task"]["id"] == "synthetic-b":
                result["samples"].pop()
            return result
        def proposer(request):
            proposal_calls.append(request)
            return demo.synthetic_proposer(request)
        with self.assertRaisesRegex(ValueError, "Incomplete response cell"):
            self.run_demo(evaluator=evaluator, proposer=proposer)
        self.assertEqual(proposal_calls, [])
        self.assertFalse((self.root / "run/selection.json").exists())

    def test_malformed_round_consumes_slots_without_replacement(self):
        requests = []
        def proposer(request):
            requests.append(request)
            return {"text": "not JSON", "provenance": {"synthetic": True}}
        result = self.run_demo(proposer=proposer)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(result["selection"]["rejections"]), 4)
        self.assertEqual(result["selection"]["selected_id"], "naive")
        self.assertEqual(len(result["fresh_display"]["summaries"]), 2)
        self.assertEqual(result["fresh_display"]["aliases"]["selected"], "naive")

    def test_all_infeasible_reports_no_winner_without_display(self):
        def evaluator(request):
            result = demo.synthetic_evaluator(request)
            if request["instruction"]["id"] != "default":
                for sample in result["samples"]:
                    sample["quality"] = 1
            return result
        result = self.run_demo(evaluator=evaluator)
        self.assertEqual(result["status"], "no_feasible_candidate")
        self.assertIsNone(result["selection"]["selected_id"])
        self.assertEqual(result["fresh_display"]["summaries"], [])
        self.assertFalse(any(demo.read(p)["phase"] == "fresh_display"
                             for p in (self.root / "run/calls").glob("*.request.json")
                             if demo.read(p)["role"] == "evaluation"))

    def test_exact_tie_retains_incumbent_and_lists_all_ties(self):
        candidates = [{"id": "naive", "count_sum": 10, "estimated_feasible": True},
                      {"id": "aaa", "count_sum": 10, "estimated_feasible": True}]
        selected, tied = demo.select(candidates, "naive")
        self.assertEqual(selected, "naive")
        self.assertEqual(tied, ["aaa", "naive"])
        self.assertEqual(demo.select(candidates)[0], "aaa")

    def test_quality_and_constraint_cannot_be_compensated_by_count(self):
        ref = {"quality": 8, "constraint_rate": 1}
        cells = [{"task": t["id"], "request_id": t["id"], "count": 5, "quality": 8,
                  "constraint_rate": .8} for t in self.plan["tasks"]]
        result = demo.summarize(self.plan["seed"], cells, ref, self.plan)
        self.assertEqual(result["count"], 5)
        self.assertFalse(result["estimated_feasible"])
        cells[0]["constraint_rate"] = cells[1]["constraint_rate"] = 1
        cells[0]["quality"] = cells[1]["quality"] = 7.49
        self.assertFalse(demo.summarize(self.plan["seed"], cells, ref, self.plan)["estimated_feasible"])

    def test_phase_seeds_are_separate_and_matched_between_arms(self):
        a = demo.evaluation_request(self.plan, self.plan["reference"], 0, "selection")
        b = demo.evaluation_request(self.plan, self.plan["seed"], 0, "selection")
        c = demo.evaluation_request(self.plan, self.plan["seed"], 0, "fresh_display")
        self.assertEqual(a["seed"], b["seed"])
        self.assertNotEqual(b["seed"], c["seed"])
        self.assertTrue(set(b["sample_ids"]).isdisjoint(c["sample_ids"]))
        self.assertEqual(a["rendered_prompt"], a["task"]["text"])
        self.assertIn("\n\nAdditional instruction:\n", b["rendered_prompt"])

    def test_duplicate_and_oversized_proposals_are_not_replaced(self):
        response = {"text": json.dumps({"instructions": ["Ｂｅ　creative.", "x" * 2100]}), "provenance": {}}
        accepted, rejected = demo.parse_proposals(response, 1, {demo.normalized("Be creative.")})
        self.assertEqual(accepted, [])
        self.assertEqual([r["reason"] for r in rejected], ["duplicate_normalized_instruction", "over_2048_serialized_utf8_bytes"])

    def test_duplicate_json_keys_and_nonfinite_values_reject_round(self):
        for raw in ['{"instructions":["a","b"],"instructions":["c","d"]}', '{"instructions":[NaN,"a"]}']:
            with self.subTest(raw=raw):
                accepted, rejected = demo.parse_proposals({"text": raw, "provenance": {}}, 1, set())
                self.assertEqual(accepted, [])
                self.assertEqual(len(rejected), 2)

    def test_plan_mutation_is_rejected_on_resume(self):
        self.run_demo()
        changed = copy.deepcopy(self.plan)
        changed["selection_seed"] += 100
        with self.assertRaisesRegex(ValueError, "immutable artifact"):
            self.run_demo(plan=changed)

    def test_completed_resume_calls_nothing(self):
        first = self.run_demo()
        def forbidden(request):
            raise AssertionError("Completed response must not be regenerated")
        second = self.run_demo(proposer=forbidden, evaluator=forbidden)
        self.assertEqual(first, second)

    def test_exact_replay_preserves_scientific_trace_and_historical_accounting(self):
        first = self.run_demo()
        plan, callback, provenance = demo.replay_callbacks(self.root / "run")
        replay = demo.run_demo(plan, self.root / "replay", callback, callback,
                              evidence_mode="replay", adapter_id="replay-v1", source_provenance=provenance)
        for key in ("rounds", "selection", "fresh_display"):
            self.assertEqual(first[key], replay[key])
        self.assertIn("zero new service calls", replay["execution"]["accounting_basis"])

    def test_tampered_saved_request_is_not_silently_used(self):
        self.run_demo()
        path = next((self.root / "run/calls").glob("*.request.json"))
        value = demo.read(path)
        value["plan_sha256"] = "wrong"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "immutable artifact"):
            self.run_demo()

    def test_tampered_response_is_rejected_before_reuse(self):
        self.run_demo()
        path = next((self.root / "run/calls").glob("*.response.json"))
        value = demo.read(path)
        value["provenance"]["changed"] = True
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "response is unbound or changed"):
            self.run_demo()

    def test_orphan_observed_response_stops_instead_of_redraw(self):
        self.run_demo()
        next((self.root / "run/calls").glob("*.response-sha256.json")).unlink()
        def forbidden(request):
            raise AssertionError("Observed response must not be regenerated")
        with self.assertRaisesRegex(ValueError, "response is unbound or changed"):
            self.run_demo(proposer=forbidden, evaluator=forbidden)

    def test_full_valid_scope_has_twenty_callbacks_and_two_hundred_initial_calls(self):
        def proposer(request):
            return {"text": json.dumps({"instructions": [f"Instruction round {request['round']} slot {i}" for i in (1, 2)]}),
                    "provenance": {}}
        result = self.run_demo(proposer=proposer)
        self.assertEqual(result["callback_requests"], 20)
        requests = [demo.read(p) for p in (self.root / "run/calls").glob("*.request.json")]
        self.assertEqual(sum(1 if r["role"] == "proposal" else 11 for r in requests), 200)

    def test_invalid_scope_and_task_support_are_rejected(self):
        for key, value in (("rounds", 3), ("display_seed", self.plan["selection_seed"]),
                           ("samples_per_cell", True), ("quality_margin", 1)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.plan)
                changed[key] = value
                with self.assertRaises(ValueError):
                    demo.validate_plan(changed)
        changed = copy.deepcopy(self.plan)
        changed["tasks"][0]["split"] = "held_out"
        with self.assertRaises(ValueError):
            demo.validate_plan(changed)


if __name__ == "__main__":
    unittest.main()
