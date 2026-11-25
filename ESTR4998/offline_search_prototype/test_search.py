"""Adversarial tests of the offline selector; synthetic temporary files only."""
import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("offline_search_under_test", Path(__file__).with_name("search.py"))
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)


class PlanTests(unittest.TestCase):
    def test_full_library_confirmation_and_check_are_reserved(self):
        plan = search.demo_plan()
        estimate = search.plan_search(plan)
        self.assertEqual(estimate["selection_evaluations"], 16)
        self.assertEqual(estimate["confirmation_evaluations_reserved"], 12)
        self.assertEqual(estimate["check_evaluations_reserved"], 4)
        self.assertEqual(estimate["maximum_evaluations"], 32)
        self.assertEqual(Decimal(estimate["maximum_reserved_cost_usd"]), Decimal("0.32"))
        self.assertIn("not API calls", estimate["unit"])

    def test_confirmation_and_check_cannot_be_squeezed_out_of_budget(self):
        for field, value in (("max_evaluations", 31), ("max_cost_usd", "0.319")):
            with self.subTest(field=field):
                plan = search.demo_plan()
                plan["budget"][field] = value
                with self.assertRaisesRegex(ValueError, "budget"):
                    search.plan_search(plan)

    def test_duplicate_candidate_id_and_rendered_instruction_are_rejected(self):
        for field in ("id", "instruction"):
            with self.subTest(field=field):
                plan = search.demo_plan()
                value = plan["candidates"][0][field]
                plan["candidates"][1][field] = "  " + value + "\n" if field == "instruction" else value
                with self.assertRaisesRegex(ValueError, "unique"):
                    search.validate_plan(plan)

    def test_internal_instruction_whitespace_is_preserved(self):
        plan = search.demo_plan()
        plan["candidates"][0]["instruction"] = "Keep  internal spacing."
        plan["candidates"][1]["instruction"] = "Keep internal spacing."
        self.assertEqual(search.validate_plan(plan)["candidates"][0]["instruction"], "Keep  internal spacing.")

    def test_task_and_seed_duplicates_are_rejected(self):
        for kind in ("task", "seed", "check"):
            with self.subTest(kind=kind):
                plan = search.demo_plan()
                if kind == "task":
                    plan["tasks"][1]["id"] = plan["tasks"][0]["id"]
                elif kind == "seed":
                    plan["selection"]["seeds"] = [10, 10]
                else:
                    plan["check_tasks"] = ["toy1", "toy1"]
                with self.assertRaisesRegex(ValueError, "unique"):
                    search.validate_plan(plan)

    def test_overlap_between_selection_and_confirmation_seeds_is_rejected(self):
        plan = search.demo_plan()
        plan["confirmation"]["seeds"] = [10, 20]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            search.validate_plan(plan)

    def test_sealed_or_unknown_task_and_same_evaluator_are_rejected(self):
        for kind in ("sealed", "unknown", "same_evaluator"):
            with self.subTest(kind=kind):
                plan = search.demo_plan()
                if kind == "sealed":
                    plan["tasks"][0]["split"] = "held_out"
                elif kind == "unknown":
                    plan["confirmation"]["tasks"].append("unknown")
                else:
                    plan["context"]["check_evaluator"] = plan["context"]["primary_evaluator"]
                with self.assertRaises(ValueError):
                    search.validate_plan(plan)

    def test_guard_with_zero_eligible_selection_tasks_is_rejected(self):
        plan = search.demo_plan()
        plan["quality_guards"][0]["eligible_tasks"] = ["toy2"]
        with self.assertRaisesRegex(ValueError, "no eligible selection"):
            search.validate_plan(plan)

    def test_boolean_and_nonfinite_plan_numbers_are_rejected(self):
        for value in (True, float("nan"), float("inf")):
            with self.subTest(value=value):
                plan = search.demo_plan()
                plan["quality_guards"][0]["margin"] = value
                with self.assertRaises(ValueError):
                    search.validate_plan(plan)

    def test_score_range_with_overflowing_width_is_rejected(self):
        plan = search.demo_plan()
        plan["quality_guards"][0]["range"] = [-1e308, 1e308]
        with self.assertRaises(ValueError):
            search.validate_plan(plan)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = search.demo_plan()
        self.serial = 0

    def run_plan(self, evaluate=search.demo_evaluator, plan=None):
        self.serial += 1
        output = self.root / f"run{self.serial}"
        current = self.plan if plan is None else plan
        report = search.run_search(current, evaluate, output, adapter_id=current["adapter_id"])
        return report, output

    def test_demo_is_complete_and_keeps_negative_confirmation_and_check(self):
        report, output = self.run_plan()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["winner"], "focused")
        self.assertFalse(report["selection"]["comparisons"]["chaotic"]["estimated_feasible"])
        self.assertEqual(report["selection"]["ranked_feasible"], ["focused", "tied"])
        self.assertEqual(report["confirmation"]["contrast"]["diversity_difference"], 0)
        self.assertLess(report["check"]["contrast"]["diversity_difference"], 0)
        self.assertEqual(report["accounting"]["attempted_evaluations"], 32)
        self.assertEqual(report["accounting"]["completed_evaluations"], 32)
        self.assertEqual(report["accounting"]["reported_underlying_requests"], 0)
        self.assertEqual(Decimal(report["accounting"]["known_cost_usd"]), Decimal("0.032"))
        self.assertEqual(json.loads((output / "report.json").read_text()), report)
        self.assertEqual(len((output / "attempts.jsonl").read_text().splitlines()), 64)

    def test_exact_replay_reproduces_scientific_fields(self):
        original, output = self.run_plan()
        replay, replay_output = self.run_plan(search.replay_records(output / "replay.json"))
        for key in ("status", "winner", "selection", "confirmation", "check", "accounting", "plan_sha256", "engine_sha256"):
            with self.subTest(key=key):
                self.assertEqual(replay[key], original[key])
        self.assertEqual(original["execution"], {"mode": "offline_callback", "source_records_sha256": None,
                                                 "accounting_basis": "callback_reported"})
        expected = {"mode": "replay", "accounting_basis": "historical_replay",
                    "source_records_sha256": hashlib.sha256((output / "replay.json").read_bytes()).hexdigest()}
        self.assertEqual(replay["execution"], expected)
        self.assertEqual(json.loads((replay_output / "manifest.json").read_text())["execution"], expected)
        original_rows = json.loads((output / "replay.json").read_text())
        self.assertEqual(original_rows, json.loads((replay_output / "replay.json").read_text()))
        self.assertTrue(all("execution" not in row["request"] for row in original_rows))

    def test_replay_source_hash_identifies_bytes_actually_loaded(self):
        _, output = self.run_plan()
        path = output / "replay.json"
        original_bytes = path.read_bytes()
        evaluator = search.replay_records(path)
        path.write_text("[]\n")
        report, _ = self.run_plan(evaluator)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["execution"]["source_records_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertNotEqual(report["execution"]["source_records_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_candidate_input_order_does_not_break_ties(self):
        plan = copy.deepcopy(self.plan)
        plan["candidates"].reverse()
        report, _ = self.run_plan(plan=plan)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["winner"], "focused")
        self.assertEqual(report["selection"]["ranked_feasible"], ["focused", "tied"])

    def test_three_seed_equal_means_use_exact_tie_and_zero_gain(self):
        plan = copy.deepcopy(self.plan)
        plan["candidates"] = [{"id": name, "instruction": f"Synthetic {name} instruction.", "source": "synthetic test"}
                              for name in ("z_second", "a_first")]
        plan["selection"] = {"tasks": ["toy0", "toy1", "toy2"], "seeds": [10, 11, 12]}
        plan["confirmation"] = {"tasks": ["toy0", "toy1", "toy2"], "seeds": [20, 21, 22]}
        plan["samples_per_cell"] = 20
        plan["budget"]["max_evaluations"] = 51
        counts_a = [[6, 6, 6], [7, 7, 7], [10, 11, 11]]
        counts_b = [[3, 3, 3], [1, 1, 1], [19, 20, 20]]

        def evaluator(request):
            reply = search.demo_evaluator(request)
            counts = counts_b if request["arm_id"] == "z_second" else counts_a
            count = counts[int(request["task_id"][-1])][request["seed"] % 10]
            for i, response in enumerate(reply["responses"]):
                response["group"] = i % count
                if request["phase"] != "check":
                    response["text"] = f"SYNTHETIC sample {response['sample_id']}"
            return reply

        report, _ = self.run_plan(evaluator, plan)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["winner"], "a_first")
        self.assertEqual(report["selection"]["ranked_feasible"], ["a_first", "z_second"])
        for arm in ("a_first", "z_second"):
            summary = report["selection"]["summaries"][arm]
            self.assertEqual(summary["diversity_count_sum"], 71)
            self.assertEqual(summary["diversity"], 71 / 9)
            comparison = report["selection"]["comparisons"][arm]
            self.assertTrue(comparison["estimated_feasible"])
            self.assertEqual(comparison["diversity_count_difference"], 0)
            self.assertEqual(comparison["diversity_difference"], 0)
            self.assertFalse(comparison["positive_observed_diversity_difference"])

    def test_manifest_preserves_original_and_normalized_instructions(self):
        plan = copy.deepcopy(self.plan)
        original = "  " + plan["candidates"][0]["instruction"] + "\n"
        plan["candidates"][0]["instruction"] = original
        report, output = self.run_plan(plan=plan)
        self.assertEqual(report["status"], "complete")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["input_plan"]["candidates"][0]["instruction"], original)
        self.assertEqual(manifest["plan"]["candidates"][0]["instruction"], original.strip())
        rows = json.loads((output / "replay.json").read_text())
        focused = next(r["request"] for r in rows if r["request"]["arm_id"] == "focused")
        self.assertEqual(focused["rendered_prompt"], focused["task"] + "\n\n" + original.strip())
        default = next(r["request"] for r in rows if r["request"]["arm_id"] == "default")
        self.assertEqual(default["rendered_prompt"], default["task"])

    def test_selection_and_confirmation_have_disjoint_sample_ids(self):
        report, output = self.run_plan()
        self.assertEqual(report["status"], "complete")
        rows = json.loads((output / "replay.json").read_text())
        ids = {phase: {sample for row in rows if row["request"]["phase"] == phase
                       for sample in row["request"]["sample_ids"]}
               for phase in ("selection", "confirmation", "check")}
        self.assertFalse(ids["selection"] & ids["confirmation"])
        self.assertTrue(ids["check"] <= ids["confirmation"])

    def test_output_cannot_be_overwritten(self):
        _, output = self.run_plan()
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            search.run_search(self.plan, search.demo_evaluator, output, adapter_id=self.plan["adapter_id"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_invalid_budget_and_adapter_fail_before_callback_or_output(self):
        for kind in ("budget", "adapter"):
            with self.subTest(kind=kind):
                plan = copy.deepcopy(self.plan)
                if kind == "budget":
                    plan["budget"]["max_evaluations"] = 31
                output = self.root / kind
                called = []
                with self.assertRaises(ValueError):
                    search.run_search(plan, lambda r: called.append(r), output,
                                      adapter_id="wrong" if kind == "adapter" else plan["adapter_id"])
                self.assertFalse(called)
                self.assertFalse(output.exists())

    def test_frozen_winner_exists_before_any_confirmation_and_is_never_rewritten(self):
        observed = []
        output = self.root / "freeze-order"

        def evaluator(request):
            if request["phase"] != "selection":
                frozen = (output / "selection.json").read_bytes()
                self.assertEqual(json.loads(frozen)["winner"], "focused")
                observed.append(frozen)
            reply = search.demo_evaluator(request)
            if request["phase"] == "confirmation" and request["arm_id"] == "focused":
                for response in reply["responses"]:
                    response["group"] = 0
                    response["quality"]["quality"] = 1.0
            return reply

        report = search.run_search(self.plan, evaluator, output, adapter_id=self.plan["adapter_id"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["winner"], "focused")
        self.assertLess(report["confirmation"]["contrast"]["diversity_difference"], 0)
        self.assertFalse(report["confirmation"]["contrast"]["estimated_feasible"])
        self.assertTrue(observed)
        self.assertTrue(all(value == (output / "selection.json").read_bytes() for value in observed))

    def test_check_judge_receives_exact_confirmed_texts_and_ids(self):
        confirmed = {}
        seen = []

        def evaluator(request):
            key = request["arm_id"], request["task_id"], request["seed"]
            if request["phase"] == "check":
                self.assertEqual(request["response_artifact"], confirmed[key])
                self.assertEqual(request["sample_ids"], [r["sample_id"] for r in confirmed[key]])
                seen.append(key)
            reply = search.demo_evaluator(request)
            if request["phase"] == "confirmation":
                confirmed[key] = [{"sample_id": r["sample_id"], "text": r["text"]} for r in reply["responses"]]
            return reply

        report, _ = self.run_plan(evaluator)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(seen), 4)

    def test_changed_check_text_is_rejected_without_reselection(self):
        def evaluator(request):
            reply = search.demo_evaluator(request)
            if request["phase"] == "check":
                reply["responses"][0]["text"] = "Different generated text."
            return reply

        report, _ = self.run_plan(evaluator)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("changed the confirmed response artifact", report["error"])
        self.assertEqual(report["winner"], "focused")
        self.assertEqual(report["accounting"]["attempted_evaluations"], 29)

    def test_all_infeasible_candidates_stop_without_confirmation(self):
        phases = []

        def evaluator(request):
            phases.append(request["phase"])
            reply = search.demo_evaluator(request)
            if request["arm_id"] != "default":
                for response in reply["responses"]:
                    response["quality"]["quality"] = 1
            return reply

        report, output = self.run_plan(evaluator)
        self.assertEqual(report["status"], "no_feasible_candidate")
        self.assertIsNone(report["winner"])
        self.assertEqual(set(phases), {"selection"})
        self.assertEqual(report["accounting"]["attempted_evaluations"], 16)
        self.assertIsNone(json.loads((output / "selection.json").read_text())["winner"])

    def test_exact_quality_margin_is_inclusive(self):
        def evaluator(request):
            reply = search.demo_evaluator(request)
            if request["arm_id"] in ("focused", "tied"):
                for response in reply["responses"]:
                    response["quality"]["quality"] = 6.5
            return reply

        report, _ = self.run_plan(evaluator)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["selection"]["comparisons"]["focused"]["estimated_feasible"])
        self.assertEqual(report["winner"], "focused")

    def test_partial_or_reordered_responses_halt_without_selecting(self):
        for kind in ("missing", "duplicate_id", "reordered"):
            with self.subTest(kind=kind):
                def evaluator(request):
                    reply = search.demo_evaluator(request)
                    if kind == "missing":
                        reply["responses"].pop()
                    elif kind == "duplicate_id":
                        reply["responses"][1]["sample_id"] = reply["responses"][0]["sample_id"]
                    else:
                        reply["responses"].reverse()
                    return reply
                report, _ = self.run_plan(evaluator)
                self.assertEqual(report["status"], "incomplete")
                self.assertIsNone(report["winner"])
                self.assertEqual(report["accounting"]["attempted_evaluations"], 1)
                self.assertEqual(Decimal(report["accounting"]["known_cost_usd"]), Decimal("0.001"))

    def test_missing_boolean_and_nonfinite_quality_never_pass(self):
        for value in (None, True, float("nan"), float("inf"), "missing"):
            with self.subTest(value=value):
                def evaluator(request):
                    reply = search.demo_evaluator(request)
                    if value == "missing":
                        reply["responses"][0]["quality"].pop("quality")
                    else:
                        reply["responses"][0]["quality"]["quality"] = value
                    return reply
                report, _ = self.run_plan(evaluator)
                self.assertEqual(report["status"], "incomplete")
                self.assertIsNone(report["winner"])
                self.assertEqual(Decimal(report["accounting"]["known_cost_usd"]), Decimal("0.001"))

    def test_nonfinite_failure_evidence_is_preserved_as_invalid_not_valid_score(self):
        def evaluator(request):
            reply = search.demo_evaluator(request)
            reply["responses"][0]["quality"]["quality"] = float("nan")
            return reply

        report, output = self.run_plan(evaluator)
        self.assertEqual(report["status"], "incomplete")
        events = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
        invalid = events[-1]["response"]["responses"][0]["quality"]["quality"]
        self.assertEqual(invalid, {"invalid_numeric": "nan"})
        self.assertEqual(json.loads((output / "replay.json").read_text()), [])

    def test_identical_responses_cannot_inflate_group_count(self):
        def evaluator(request):
            reply = search.demo_evaluator(request)
            reply["responses"][1]["text"] = reply["responses"][0]["text"]
            reply["responses"][1]["group"] = reply["responses"][0]["group"] + 1
            return reply

        report, _ = self.run_plan(evaluator)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("identical responses must share a group", report["error"])

    def test_mismatched_result_identity_is_rejected(self):
        for field in ("request_id", "generator", "evaluator", "metric_version"):
            with self.subTest(field=field):
                def evaluator(request):
                    reply = search.demo_evaluator(request)
                    reply[field] = "different"
                    return reply
                report, _ = self.run_plan(evaluator)
                self.assertEqual(report["status"], "incomplete")
                self.assertIn(field, report["error"])

    def test_callback_exception_is_an_attempt_with_unknown_reserved_cost(self):
        called = []

        def evaluator(request):
            called.append(request)
            raise RuntimeError("synthetic adapter failure")

        report, output = self.run_plan(evaluator)
        self.assertEqual(len(called), 1)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["accounting"]["attempted_evaluations"], 1)
        self.assertEqual(report["accounting"]["unknown_cost_attempts"], 1)
        self.assertEqual(report["accounting"]["unknown_request_usage_attempts"], 1)
        self.assertEqual(Decimal(report["accounting"]["unknown_reserved_usd"]), Decimal("0.01"))
        events = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual([e["status"] for e in events], ["pending", "failed"])

    def test_unknown_invalid_or_excess_cost_stops_new_callbacks(self):
        for cost in (None, "-1", "NaN", True, "0.02"):
            with self.subTest(cost=cost):
                def evaluator(request):
                    reply = search.demo_evaluator(request)
                    reply["usage"]["cost_usd"] = cost
                    return reply
                report, _ = self.run_plan(evaluator)
                self.assertEqual(report["status"], "incomplete")
                self.assertEqual(report["accounting"]["attempted_evaluations"], 1)
                if cost == "0.02":
                    self.assertEqual(Decimal(report["accounting"]["known_cost_usd"]), Decimal("0.02"))
                else:
                    self.assertEqual(report["accounting"]["unknown_cost_attempts"], 1)

    def test_malformed_judgment_retains_known_cost_and_underlying_requests(self):
        def evaluator(request):
            reply = search.demo_evaluator(request)
            reply["usage"]["underlying_requests"] = 7
            reply["responses"].pop()
            return reply

        report, _ = self.run_plan(evaluator)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(Decimal(report["accounting"]["known_cost_usd"]), Decimal("0.001"))
        self.assertEqual(report["accounting"]["reported_underlying_requests"], 7)

    def test_confirmation_exception_keeps_frozen_winner_without_runner_up(self):
        phases = []

        def evaluator(request):
            phases.append(request["phase"])
            if request["phase"] == "confirmation":
                raise RuntimeError("synthetic confirmation failure")
            return search.demo_evaluator(request)

        report, output = self.run_plan(evaluator)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["winner"], "focused")
        self.assertEqual(json.loads((output / "selection.json").read_text())["winner"], "focused")
        self.assertEqual(phases.count("confirmation"), 1)
        self.assertNotIn("check", phases)

    def test_no_eligible_check_guard_remains_unavailable(self):
        plan = copy.deepcopy(self.plan)
        plan["check_tasks"] = ["toy0"]
        report, _ = self.run_plan(plan=plan)
        self.assertEqual(report["status"], "complete")
        guard = report["check"]["contrast"]["guards"]["constraint"]
        self.assertEqual(guard["tasks"], 0)
        self.assertIsNone(guard["difference"])
        self.assertIsNone(guard["point_guard_pass"])
        self.assertIsNone(report["check"]["contrast"]["estimated_feasible"])

    def test_callback_cannot_mutate_plan_or_engine_request(self):
        before = copy.deepcopy(self.plan)

        def evaluator(request):
            reply = search.demo_evaluator(request)
            request["decoding"]["max_output_tokens"] = 9999
            request["quality_guards"][0]["margin"] = 9999
            return reply

        report, output = self.run_plan(evaluator)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(self.plan, before)
        rows = json.loads((output / "replay.json").read_text())
        self.assertTrue(all(r["request"]["decoding"]["max_output_tokens"] == 128 for r in rows))

    def test_changed_plan_missing_record_and_duplicate_replay_are_rejected(self):
        _, output = self.run_plan()
        rows = json.loads((output / "replay.json").read_text())
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(json.dumps([rows[0], rows[0]]))
        with self.assertRaisesRegex(ValueError, "duplicate replay"):
            search.replay_records(duplicate)
        incomplete = self.root / "incomplete.json"
        incomplete.write_text(json.dumps(rows[1:]))
        result, _ = self.run_plan(search.replay_records(incomplete))
        self.assertEqual(result["status"], "incomplete")
        changed = copy.deepcopy(self.plan)
        changed["candidates"][0]["instruction"] += " Changed."
        result, _ = self.run_plan(search.replay_records(output / "replay.json"), changed)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("no exact replay record", result["error"])

    def test_modified_request_without_rehash_is_rejected(self):
        _, output = self.run_plan()
        rows = json.loads((output / "replay.json").read_text())
        rows[0]["request"]["seed"] += 1
        changed = self.root / "modified.json"
        changed.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            search.replay_records(changed)


class MathematicsTests(unittest.TestCase):
    def test_arithmetic_tolerance_cannot_accept_full_tiny_scale_quality_loss(self):
        plan = search.demo_plan()
        plan["quality_guards"] = [{"measure": "quality", "margin": 0,
                                   "range": [0, 1e-12], "eligible_tasks": ["toy0"]}]
        candidate = {"by_task": {"toy0": {"diversity": 3, "quality": {"quality": 0}}},
                     "tasks": 1, "repetitions": 1, "diversity_count_sum": 3}
        reference = {"by_task": {"toy0": {"diversity": 2, "quality": {"quality": 1e-12}}},
                     "tasks": 1, "repetitions": 1, "diversity_count_sum": 2}
        result = search.contrast(plan, candidate, reference)
        self.assertEqual(result["guards"]["quality"]["difference"], -1e-12)
        self.assertFalse(result["guards"]["quality"]["point_guard_pass"])
        self.assertFalse(result["estimated_feasible"])
        self.assertLess(result["guards"]["quality"]["boundary_tolerance"], 1e-12)

    def test_exact_count_contrast_preserves_small_real_gain_and_loss(self):
        plan = search.demo_plan()
        cells = [{"task_id": f"toy{t}", "seed": r, "diversity": 2,
                  "quality": {"quality": 7, **({"constraint": 1} if t == 1 else {})}}
                 for t in range(3) for r in range(3)]
        reference = search.summarize(plan, cells, ["toy0", "toy1", "toy2"], [0, 1, 2])
        cells[-1]["diversity"] += 1
        candidate = search.summarize(plan, cells, ["toy0", "toy1", "toy2"], [0, 1, 2])
        gain = search.contrast(plan, candidate, reference)
        loss = search.contrast(plan, reference, candidate)
        self.assertEqual(gain["diversity_count_difference"], 1)
        self.assertEqual(gain["diversity_difference"], 1 / 9)
        self.assertTrue(gain["positive_observed_diversity_difference"])
        self.assertEqual(loss["diversity_count_difference"], -1)
        self.assertEqual(loss["diversity_difference"], -1 / 9)
        self.assertFalse(loss["positive_observed_diversity_difference"])
        mismatched = search.summarize(plan, [c for c in cells if c["seed"] != 2],
                                     ["toy0", "toy1", "toy2"], [0, 1])
        with self.assertRaisesRegex(ValueError, "equal complete grids"):
            search.contrast(plan, candidate, mismatched)

    def test_repetitions_are_averaged_within_task_and_guards_use_eligible_tasks(self):
        plan = search.demo_plan()
        cells = [
            {"task_id": "toy0", "seed": 10, "diversity": 1, "quality": {"quality": 3}},
            {"task_id": "toy0", "seed": 11, "diversity": 3, "quality": {"quality": 5}},
            {"task_id": "toy1", "seed": 10, "diversity": 4, "quality": {"quality": 8, "constraint": 0.25}},
            {"task_id": "toy1", "seed": 11, "diversity": 4, "quality": {"quality": 10, "constraint": 0.75}},
        ]
        result = search.summarize(plan, cells, ["toy0", "toy1"], [10, 11])
        self.assertEqual(result["by_task"]["toy0"]["diversity"], 2)
        self.assertEqual(result["diversity"], 3)
        self.assertEqual(result["quality"]["quality"], {"mean": 6.5, "tasks": 2})
        self.assertEqual(result["quality"]["constraint"], {"mean": 0.5, "tasks": 1})
        for malformed in (cells[:-1], cells + [cells[0]]):
            with self.assertRaises(ValueError):
                search.summarize(plan, malformed, ["toy0", "toy1"], [10, 11])


if __name__ == "__main__":
    unittest.main()
