"""Offline selection binding, readiness accounting and second-model replay tests."""
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import second_generator_experiment as stage
import api_runner as op
import api_data as data
import api_runtime as runtime
import test_factorial_experiment as fixtures
FakeAPI = fixtures.FakeAPI


class SecondGeneratorTest(unittest.TestCase):
    def setUp(self):
        # Reuse only the synthetic fixture factory, never its completed-run files.
        fixtures.FactorialTest.setUp(self)
        patcher = patch.object(stage, 'SOURCE_HASHES', {
            name: data.sha(self.source / name) for name in ('manifest.json', 'results.json', 'report.json')})
        patcher.start(); self.addCleanup(patcher.stop)
        original = json.loads((self.root / 'generator-snapshot.json').read_text())
        selected = copy.deepcopy(original)
        selected['model'] = selected['endpoint']['model_id'] = 'qwen/synthetic-second-model'
        self.snapshot = self.root / 'selected-endpoint.json'
        op.write_json(self.snapshot, selected)
        self.selection = self.root / 'selection.json'
        op.write_json(self.selection, {'schema': stage.SELECTION_SCHEMA,
            'decision_reference': 'Synthetic prospective selection', 'generator': {
                'label': 'anchor', 'model': selected['model'], 'provider': selected['provider'],
                'family': 'qwen', 'reserve_usd': .02, 'endpoint_snapshot': str(self.snapshot)}})
        self.candidate_path = self.root / 'candidate.json'
        self.candidate = stage.build_candidate_plan(self.source, self.selection)
        op.write_json(self.candidate_path, self.candidate)
        self.preparation = self.root / 'preparation'
        self.preparation.mkdir()
        self.preparation_file = self.preparation / 'receipt.json'
        op.write_json(self.preparation_file, {'passed': True, 'complete': True, 'live': False, 'accounting': {
            'attempted_requests': 23, 'known_cost_usd': .125, 'committed_usd': .125,
            'unknown_cost_attempts': 0}})

        def fake_verifier(directory, candidate):
            self.assertEqual(Path(directory), self.preparation)
            self.assertEqual(candidate, self.candidate)
            value = json.loads(self.preparation_file.read_text())
            return value | {'files_sha256': {str(self.preparation_file): data.sha(self.preparation_file)}}
        module = types.ModuleType('second_generator_readiness')
        module.verify_receipt = fake_verifier
        patcher = patch.dict(sys.modules, {'second_generator_readiness': module})
        patcher.start(); self.addCleanup(patcher.stop)
        self.plan = stage.build_plan(self.candidate_path, self.preparation)
        self.plan_path = self.root / 'final.json'
        op.write_json(self.plan_path, self.plan)

    def compact(self):
        plan = copy.deepcopy(self.plan)
        plan['cells'] = [c for c in plan['cells'] if c['task'] == 'narrative-dev-01'][:2]
        cells = {op.fingerprint(c) for c in plan['cells']}
        plan['jobs'] = [j for j in plan['jobs'] if op.fingerprint(j['metadata']['cell']) in cells]
        plan['initial_requests'] = len(plan['jobs'])
        op.write_json(self.plan_path, plan)
        patcher = patch.object(stage, 'verify_plan', return_value=plan)
        patcher.start(); self.addCleanup(patcher.stop)
        return plan

    def test_bound_model_change_retains_design_and_all_other_settings(self):
        self.assertEqual(stage.verify_candidate_plan(self.candidate_path), self.candidate)
        self.assertEqual(stage.verify_plan(self.plan_path), self.plan)
        self.assertEqual((len(self.plan['jobs']), len(self.plan['cells']), len(self.plan['arms'])), (4554, 180, 9))
        self.assertEqual(self.plan['seed'], 20261117)
        self.assertEqual(self.plan['analysis']['bootstrap_seed'], 20261116)
        self.assertEqual(self.plan['combined_initial_requests'], 4577)
        for job in self.plan['jobs']:
            self.assertEqual(job['metadata']['phase'], stage.SCHEMA)
            if job['route'] == 'generator':
                body = job['request']
                self.assertEqual(body['model'], 'qwen/synthetic-second-model')
                self.assertEqual({k: body[k] for k in self.generation if k not in ('seed', 'temperature')},
                    {k: v for k, v in self.generation.items() if k not in ('seed', 'temperature')})
        for path in (self.selection, self.snapshot, self.candidate_path, self.preparation_file):
            self.assertEqual(self.plan['source_files_sha256'][str(path)], data.sha(path))

    def test_candidate_cannot_execute_before_readiness(self):
        fake = FakeAPI()
        with self.assertRaisesRegex(ValueError, 'passing readiness'):
            stage.execute(self.candidate_path, self.output, fake)
        self.assertEqual(fake.calls, [])
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, 'passing live preparation'):
            stage.execute(self.plan_path, self.output)

    def test_selection_or_snapshot_edits_invalidate_candidate(self):
        selection = json.loads(self.selection.read_text())
        selection['decision_reference'] += ' changed after freeze'
        op.write_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, 'bound source changed'):
            stage.verify_candidate_plan(self.candidate_path)

    def test_no_silent_parameter_drop_or_same_family_judge(self):
        saved = json.loads(self.snapshot.read_text())
        changed = copy.deepcopy(saved)
        changed['endpoint']['supported_parameters'].remove('min_p')
        op.write_json(self.snapshot, changed)
        with self.assertRaisesRegex(ValueError, 'Unsupported parameters'):
            stage.build_candidate_plan(self.source, self.selection)
        op.write_json(self.snapshot, saved)
        selection = json.loads(self.selection.read_text())
        selection['generator']['family'] = 'google'
        op.write_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, 'identity/family'):
            stage.build_candidate_plan(self.source, self.selection)

    def test_failed_unknown_or_changed_readiness_rejected(self):
        original = json.loads(self.preparation_file.read_text())
        for bad in (original | {'passed': False}, original | {'accounting': original['accounting'] |
                     {'unknown_cost_attempts': 1}}, original | {'accounting': original['accounting'] |
                     {'attempted_requests': 47}}):
            op.write_json(self.preparation_file, bad)
            with self.assertRaisesRegex(ValueError, 'Readiness must be complete'):
                stage.build_plan(self.candidate_path, self.preparation)
        op.write_json(self.preparation_file, original | {'accounting': original['accounting'] |
            {'known_cost_usd': .15, 'committed_usd': .15}})
        with self.assertRaisesRegex(ValueError, 'preparation evidence'):
            stage.verify_plan(self.plan_path)

    def test_combined_budget_is_not_reset_and_resume_reuses_calls(self):
        self.assertEqual(self.plan['limits']['max_cost_usd'], 19.875)
        self.assertEqual(self.plan['combined_limits']['max_requests'], 9154)
        plan = self.compact()
        first = FakeAPI(invalid_judge=True)
        result = stage.execute(self.plan_path, self.output, first)
        self.assertTrue(result['complete'])
        self.assertEqual(result['combined_accounting']['attempted_requests'], len(first.calls) + 23)
        self.assertAlmostEqual(result['combined_accounting']['known_cost_usd'], len(first.calls) * .001 + .125)
        second = FakeAPI()
        self.assertEqual(stage.execute(self.plan_path, self.output, second), result)
        self.assertEqual(second.calls, [])

    def test_combined_cost_stop_precedes_first_main_submission(self):
        plan = self.compact()
        plan['readiness']['accounting'].update(known_cost_usd=19.995, committed_usd=19.995)
        plan['limits']['max_cost_usd'] = .005
        op.write_json(self.plan_path, plan)
        client = FakeAPI()
        with self.assertRaisesRegex(ValueError, 'Cost budget would be exceeded'):
            stage.execute(self.plan_path, self.output, client)
        self.assertEqual(client.calls, [])
        result = json.loads((self.output / 'results.json').read_text())
        self.assertEqual(result['combined_accounting']['attempted_requests'], 23)
        self.assertEqual(result['combined_accounting']['known_cost_usd'], 19.995)

    def test_missing_or_nonzero_reasoning_stops_without_regeneration(self):
        self.compact()
        for count in (None, 1):
            class BadReasoning(FakeAPI):
                def complete(self, body):
                    raw = super().complete(body)
                    raw['usage']['completion_tokens_details']['reasoning_tokens'] = count
                    return raw
            output = self.root / ('bad-reasoning-' + str(count))
            first = BadReasoning()
            with self.assertRaisesRegex(ValueError, 'Generation control failure'):
                stage.execute(self.plan_path, output, first)
            self.assertEqual(len(first.calls), 1)
            saved = json.loads((output / 'results.json').read_text())
            self.assertFalse(saved['complete'])
            self.assertEqual(len(saved['generation_control_failures']), 1)
            records = list((output / 'requests').glob('*.json'))
            self.assertEqual(len(records), 1)
            before = data.sha(records[0])
            self.assertEqual(json.loads(records[0].read_text())['attempts'][0]['status'], 'ok')
            second = FakeAPI()
            with self.assertRaisesRegex(ValueError, 'Saved generation control failure'):
                stage.execute(self.plan_path, output, second)
            self.assertEqual(second.calls, [])
            self.assertEqual(data.sha(records[0]), before)


if __name__ == '__main__':
    unittest.main()
