"""Offline behavioral tests for the explicit post-observation amendment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import workflow as workflow

base = workflow.base


class ContinuationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = base.synthetic_plan()
        self.parent = self.root / "parent"
        self.trace = self.parent / "trace"
        self.output = self.root / "continuation-run"
        self.requests = []

        def proposer(request):
            return {"text": json.dumps({"instructions": [f"Round {request['round']} instruction {i}." for i in (1, 2)]}),
                    "provenance": {"synthetic": True}}

        def original_evaluator(request):
            if request["instruction"]["id"] == "proposal-r2-s1" and request["task"]["id"] == self.plan["tasks"][1]["id"]:
                raise ValueError("original observed length: fixed protocol stops")
            return self.response(request)

        with self.assertRaisesRegex(ValueError, "original observed length"):
            base.run_demo(self.plan, self.trace, proposer, original_evaluator,
                          evidence_mode="synthetic", adapter_id="fixture-v1")
        interrupted = next(base.read(p) for p in (self.trace / "calls").glob("*.request.json")
                           if not p.with_name(p.name.replace(".request.", ".response.")).exists())
        receipt = self.length_receipt(interrupted)
        raw = self.parent / "api/requests" / (receipt["api_request_id"] + ".json")
        raw.parent.mkdir(parents=True)
        raw.write_text(json.dumps({"synthetic": True, "finish_reason": "length"}) + "\n")
        receipt["request_file_sha256"] = base.file_sha(raw)
        plan_path = self.root / "original-plan.json"
        base.save_new(plan_path, self.plan)
        self.amendment = {"schema": workflow.SCHEMA,
            "parent": {"path": str(self.parent), "files_sha256": {
                str(p.relative_to(self.parent)): base.file_sha(p) for p in self.parent.rglob("*.json")},
                       "accounting": {"attempted_requests": 105, "known_cost_usd": .12}, "initial_request_ids": 105},
            "original_plan": {"path": str(plan_path), "sha256": base.file_sha(plan_path)},
            "parent_truncation": receipt, "code_sha256": {},
            "policy": {"excluded_candidate": "proposal-r2-s1"},
            "limits": {"max_initial_calls": 200, "max_attempts": 400, "max_cost_usd": 5,
                       "max_new_initial_calls": 88, "max_new_attempts": 176}}

    def response(self, request, *, count=None, quality=None):
        identity = request["instruction"]["id"]
        count = count if count is not None else (3 if identity == "default" else 5 if identity in {"proposal-r1-s2", "proposal-r2-s2"} else 4)
        quality = quality if quality is not None else (1 if identity == "proposal-r1-s1" else 8)
        return {"samples": [{"id": identity_, "text": "Synthetic response " + identity_, "group": i % count,
                             "quality": quality, "constraint_pass": True} for i, identity_ in enumerate(request["sample_ids"])],
                "provenance": {"synthetic": True}}

    def length_receipt(self, request):
        return {"schema": "estr-continuation-length-v1", "request_id": request["request_id"],
                "api_request_id": request["request_id"].replace("demo-", "pd-") + "-g3",
                "request_file_sha256": "0" * 64, "attempt": 0, "finish_reason": "length", "reasoning_tokens": 0,
                "known_cost_usd": .001, "observed_text_nonempty": True, "other_controls_passed": True}

    def truncate(self, request, **updates):
        receipt = self.length_receipt(request) | updates
        raise workflow.CellTruncated(request["request_id"], receipt["api_request_id"], receipt)

    def evaluate(self, request):
        self.requests.append(copy.deepcopy(request))
        return self.response(request)

    def run_workflow(self, evaluator=None, output=None):
        return workflow.run_continuation(self.plan, self.trace, output or self.output,
                                         evaluator or self.evaluate, amendment=self.amendment,
                                         evidence_mode="synthetic", adapter_id="fixture-v1")

    def test_reconstructs_exact_history_excludes_whole_candidate_and_limits_new_scope(self):
        before = {str(p): base.file_sha(p) for p in self.parent.rglob("*") if p.is_file()}
        report = self.run_workflow()
        self.assertTrue(report["workflow_complete"])
        self.assertTrue(report["fresh_display_complete"])
        self.assertFalse(report["original_plan_complete"])
        self.assertTrue(report["post_observation_amendment"])
        self.assertFalse(report["effectiveness_claim"])
        self.assertEqual(report["new_proposal_calls"], 0)
        self.assertEqual(len(self.requests), 8)
        self.assertTrue(all(r["role"] == "evaluation" for r in self.requests))
        self.assertEqual([r["instruction"]["id"] for r in self.requests if r["phase"] == "selection"],
                         ["proposal-r2-s2", "proposal-r2-s2"])
        self.assertNotIn("proposal-r2-s1", [c["id"] for c in report["selection"]["candidates"]])
        exclusion = report["selection"]["exclusions"][0]
        self.assertEqual(len(exclusion["retained_complete_cells"]), 1)
        self.assertNotIn("count", exclusion)
        self.assertEqual(report["selection"]["selected_id"], "proposal-r1-s2")
        self.assertEqual(report["selection"]["tied_best_ids"], ["proposal-r1-s2", "proposal-r2-s2"])
        self.assertEqual(before, {str(p): base.file_sha(p) for p in self.parent.rglob("*") if p.is_file()})
        for p in (self.trace / "calls").glob("*.json"):
            self.assertEqual(p.read_bytes(), (self.output / "trace/calls" / p.name).read_bytes())

    def test_selection_frozen_before_display_and_adverse_display_does_not_reselect(self):
        def evaluator(request):
            if request["phase"] == "fresh_display":
                selected = base.read(self.output / "continuation/selection.json")
                self.assertEqual(selected["selected_id"], "proposal-r1-s2")
                if request["instruction"]["id"] == "proposal-r1-s2":
                    return self.response(request, count=1, quality=1)
            return self.response(request)
        result = self.run_workflow(evaluator)
        self.assertEqual(result["selection"]["selected_id"], "proposal-r1-s2")
        fresh = next(c for c in result["fresh_display"]["summaries"] if c["id"] == "proposal-r1-s2")
        self.assertEqual(fresh["count"], 1)
        self.assertFalse(fresh["estimated_feasible"])

    def test_new_selection_length_excludes_entire_candidate_and_skips_its_other_task(self):
        calls = []
        def evaluator(request):
            calls.append(request)
            if request["phase"] == "selection":
                self.truncate(request)
            return self.response(request)
        report = self.run_workflow(evaluator)
        self.assertEqual(len([r for r in calls if r["phase"] == "selection"]), 1)
        self.assertEqual([x["id"] for x in report["selection"]["exclusions"]], ["proposal-r2-s1", "proposal-r2-s2"])
        self.assertNotIn("proposal-r2-s2", [c["id"] for c in report["selection"]["candidates"]])
        self.assertTrue(report["fresh_display_complete"])

    def test_missing_default_display_has_no_aggregate_or_feasibility_comparisons(self):
        calls = []
        def evaluator(request):
            calls.append(request)
            if request["phase"] == "fresh_display" and request["instruction"]["id"] == "default" and request["task"] == self.plan["tasks"][0]:
                self.truncate(request)
            return self.response(request)
        report = self.run_workflow(evaluator)
        self.assertEqual(len(calls), 8)
        self.assertEqual(report["status"], "workflow_complete_with_missing_display")
        self.assertFalse(report["fresh_display_complete"])
        self.assertFalse(report["fresh_display"]["comparison_available"])
        self.assertEqual(len(report["fresh_display"]["cells"]), 6)
        self.assertEqual(sum(c["status"] == "complete" for c in report["fresh_display"]["cells"]), 5)
        self.assertNotIn("default", [c["id"] for c in report["fresh_display"]["summaries"]])
        for c in report["fresh_display"]["summaries"]:
            self.assertIsNone(c["quality_difference"])
            self.assertIsNone(c["constraint_difference"])
            self.assertIsNone(c["estimated_feasible"])
            self.assertFalse(c["is_reference"])

    def test_selected_display_missing_cell_never_produces_partial_aggregate_or_new_winner(self):
        def evaluator(request):
            if request["phase"] == "fresh_display" and request["instruction"]["id"] == "proposal-r1-s2" and request["task"] == self.plan["tasks"][1]:
                self.truncate(request)
            return self.response(request)
        report = self.run_workflow(evaluator)
        self.assertEqual(report["selection"]["selected_id"], "proposal-r1-s2")
        self.assertFalse(report["fresh_display_complete"])
        self.assertNotIn("proposal-r1-s2", [c["id"] for c in report["fresh_display"]["summaries"]])
        missing = next(c for c in report["fresh_display"]["cells"] if c["status"] != "complete")
        self.assertIsNone(missing["summary"])
        self.assertTrue(workflow.verify_replay(self.output)["verified"])

    def test_completed_resume_calls_nothing_and_replay_preserves_scientific_artifacts(self):
        report = self.run_workflow()
        def forbidden(request):
            raise AssertionError("Saved observations must not be repeated")
        self.assertEqual(report, self.run_workflow(forbidden))
        replay = workflow.replay_continuation(self.output, self.root / "replay")
        for key in ("original_rounds", "selection", "fresh_display", "fresh_display_complete"):
            self.assertEqual(report[key], replay[key])
        self.assertIn("zero new service calls", replay["execution"]["accounting_basis"])

    def test_observed_truncation_is_not_regenerated_on_resume(self):
        count = 0
        def evaluator(request):
            nonlocal count
            if request["phase"] == "selection":
                count += 1
                self.truncate(request)
            return self.response(request)
        first = self.run_workflow(evaluator)
        self.assertEqual(first, self.run_workflow(evaluator))
        self.assertEqual(count, 1)

    def test_empty_malformed_billing_and_routing_faults_stop_instead_of_skip(self):
        for number, error in enumerate((ValueError("empty output"), ValueError("malformed judgment"),
                                        RuntimeError("unknown billing"), RuntimeError("wrong route"))):
            with self.subTest(error=str(error)):
                output = self.root / ("fault-" + str(number))
                def evaluator(request):
                    raise error
                with self.assertRaises(type(error)):
                    self.run_workflow(evaluator, output)
                self.assertFalse((output / "continuation/selection.json").exists())
                self.assertFalse((output / "continuation/report.json").exists())

    def test_invalid_length_certificates_do_not_enable_skipping(self):
        for number, changes in enumerate(({"known_cost_usd": float("nan")}, {"other_controls_passed": False},
                                          {"observed_text_nonempty": False}, {"reasoning_tokens": 1},
                                          {"finish_reason": "stop"}, {"attempt": 2})):
            with self.subTest(changes=changes):
                def evaluator(request):
                    self.truncate(request, **changes)
                with self.assertRaises(ValueError):
                    self.run_workflow(evaluator, self.root / ("invalid-" + str(number)))

    def test_invalid_observed_evaluation_remains_invalid_without_redraw(self):
        calls = 0
        def evaluator(request):
            nonlocal calls
            calls += 1
            response = self.response(request)
            response["samples"].pop()
            return response
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "Incomplete response cell"):
                self.run_workflow(evaluator)
        self.assertEqual(calls, 1)

    def test_parent_mutation_or_feedback_mismatch_stops_before_new_calls(self):
        target = self.trace / "rounds/round-1-selection.json"
        data = base.read(target)
        data["incumbent_after"] = "naive"
        target.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Frozen parent evidence differs"):
            self.run_workflow()
        self.amendment["parent"]["files_sha256"][str(target.relative_to(self.parent))] = base.file_sha(target)
        with self.assertRaisesRegex(ValueError, "Original round-one selection differs"):
            self.run_workflow()
        self.assertEqual(self.requests, [])

    def test_original_plan_does_not_get_amended_or_new_request_ids(self):
        self.run_workflow()
        for request in self.requests:
            index = [t["id"] for t in self.plan["tasks"]].index(request["task"]["id"])
            self.assertEqual(request, base.evaluation_request(self.plan, request["instruction"], index, request["phase"]))
        stored = base.read(self.output / "continuation/selection.json")
        self.assertEqual(stored["original_plan_sha256"], base.digest(self.plan))
        self.assertEqual(stored["amendment_sha256"], base.digest(self.amendment))

    def test_tampered_callback_or_presentation_is_rejected_on_resume_or_replay(self):
        self.run_workflow()
        target = self.output / "continuation/trace.md"
        original = target.read_text()
        target.write_text(original + "Misleading extra claim.\n")
        with self.assertRaisesRegex(ValueError, "presentation trace differs"):
            workflow.verify_replay(self.output)
        target.write_text(original)
        request = next(r for r in self.requests if r["phase"] == "selection")
        response_path = self.output / "trace/calls" / (request["request_id"] + ".response.json")
        response = base.read(response_path)
        response["samples"][0]["quality"] = 1
        response_path.write_text(json.dumps(response))
        with self.assertRaisesRegex(ValueError, "unbound or changed"):
            self.run_workflow()

    def test_replay_checks_completion_and_claim_metadata_not_only_scores(self):
        self.run_workflow()
        target = self.output / "continuation/report.json"
        report = base.read(target)
        report["effectiveness_claim"] = True
        target.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "does not reproduce original effectiveness_claim"):
            workflow.verify_replay(self.output)


if __name__ == "__main__":
    unittest.main()
