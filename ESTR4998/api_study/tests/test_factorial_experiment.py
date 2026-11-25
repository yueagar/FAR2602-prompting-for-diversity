"""Synthetic offline tests: no credentials, network, or original run mutations."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import factorial_experiment as stage
import api_runner as op
import api_data as data
import api_evaluation as ev
import api_runtime as runtime


class FakeAPI:
    def __init__(self, unknown_at=None, invalid_judge=False, bad_generation=False, alternate_groups=False):
        self.calls = []
        self.unknown_at, self.invalid_judge = unknown_at, invalid_judge
        self.bad_generation = bad_generation
        self.alternate_groups, self.group_bodies = alternate_groups, {}

    def complete(self, body):
        self.calls.append(copy.deepcopy(body))
        role = body.get('response_format', {}).get('json_schema', {}).get('name', 'generation')
        if role == 'generation':
            text = ('Synthetic response ' + op.fingerprint(body)[:8] if self.alternate_groups
                    else 'Repeated synthetic response.')
            if self.bad_generation and len(self.calls) == 1:
                text = None
        elif self.invalid_judge:
            self.invalid_judge = False
            text = 'invalid judgment'
        elif role == 'grouping':
            # A non-singleton partition checks inversion of the saved anonymous order.
            identity = op.fingerprint(body)
            seen = self.group_bodies.get(identity, 0)
            self.group_bodies[identity] = seen + 1
            groups = ([[i] for i in range(10)] if self.alternate_groups and seen
                      else [[0, 1]] + [[i] for i in range(2, 10)])
            text = json.dumps({'groups': groups})
        else:
            text = '{"score": 8}'
        return {'id': f'synthetic-{len(self.calls)}', 'model': body['model'],
            'provider': body['provider']['only'][0],
            'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
            'usage': {'cost': None if len(self.calls) == self.unknown_at else .001,
                      'completion_tokens_details': {'reasoning_tokens': 0}}}


class FactorialTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'frozen-main'
        self.source.mkdir()
        specs = [('generator', 'anchor', 'deepseek/deepseek-v4.1-flash-1017', 'deepinfra/fp8', 'deepseek'),
                 ('primary', 'gemini', 'google/gemini-4-hydrogen', 'google-ai-studio/flex', 'google'),
                 ('check', 'gpt', 'openai/gpt-6-luna', 'openai', 'openai')]
        routes, snapshots = {}, {}
        for role, label, model, provider, family in specs:
            path = self.root / (role + '-snapshot.json')
            route = {'label': label, 'model': model, 'provider': provider, 'family': family,
                     'endpoint_snapshot': str(path), 'reserve_usd': .02}
            if role != 'generator':
                route.update(json_schema=True, parameters={'max_tokens': 4096, 'reasoning': {'effort': 'low'}})
            op.write_json(path, {'model': model, 'provider': provider, 'endpoint': {
                'tag': provider, 'model_id': model, 'provider_name': provider,
                'context_length': 1048576, 'max_completion_tokens': 90000,
                'supported_parameters': list(op.PARAMETERS) + ['structured_outputs', 'response_format'],
                'pricing': {'prompt': '0.00000001', 'completion': '0.00000001'}}})
            routes[role] = route
            snapshots['g-anchor-default' if role == 'generator' else 'j-' + label] = {
                'path': str(path), 'sha256': data.sha(path)}
        arms = [{'id': name, 'instruction': text, 'decoding_overrides': {}} for name, text in [
            ('default', ''), ('naive_creative', 'Generate a creative and unusual response.'),
            ('manual_causal_shift_v1', 'Synthetic causal instruction.'),
            ('manual_alternative_selection_v1', 'Unused instruction.')]]
        self.generation = {'max_tokens': 1800, 'min_p': 0, 'reasoning': {'effort': 'none'},
                           'seed': 20261114, 'temperature': .8, 'top_k': 0, 'top_p': .95}
        cfg = {'arms': arms, 'repetitions': 2, 'generators': [routes['generator']],
               'judges': [routes['primary'], routes['check']], 'generation': self.generation,
               'evaluation_profile': ev.TASK_SENSITIVE_PROFILE}
        op.write_json(self.source / 'manifest.json', {'git_commit': 'original-freeze', 'specification': {
            'phase': 'main', 'n': 10, 'config': cfg,
            'tasks': [{'id': task, 'text': 'Synthetic development task ' + task} for task in stage.TASKS],
            'check_tasks': {'anchor': ['narrative-dev-01', 'curated-18', 'narrative-dev-02']},
            'snapshots': snapshots}})
        op.write_json(self.source / 'results.json', {'phase': 'main', 'exploratory': True,
            'confirmatory': False, 'main_eligible': False,
            'accounting': {'known_cost_usd': 1, 'unknown_cost_attempts': 0}})
        op.write_json(self.source / 'report.json', {'frozen': True})
        patched = patch.object(stage, 'SOURCE_HASHES', {name: data.sha(self.source / name)
            for name in ('manifest.json', 'results.json', 'report.json')})
        patched.start(); self.addCleanup(patched.stop)
        patched = patch.object(runtime, 'code_hashes', return_value={'synthetic-code': 'fixed'})
        patched.start(); self.addCleanup(patched.stop)
        self.plan = stage.build_plan(self.source)
        self.plan_path = self.root / 'plan.json'
        op.write_json(self.plan_path, self.plan)
        self.output = self.root / 'run'

    def compact(self, retain_roster=False):
        """Exercise the real executor on two complete synthetic cells for speed."""
        plan = copy.deepcopy(self.plan)
        cells = [c for c in plan['cells'] if c['task'] == 'narrative-dev-01'][:2]
        selected = {op.fingerprint(c) for c in cells}
        plan['jobs'] = [j for j in plan['jobs'] if op.fingerprint(j['metadata']['cell']) in selected]
        if not retain_roster:
            plan['cells'] = cells
        plan['initial_requests'] = len(plan['jobs'])
        op.write_json(self.plan_path, plan)
        patched = patch.object(stage, 'verify_plan', return_value=plan)
        patched.start(); self.addCleanup(patched.stop)
        return plan

    def test_exact_scope_settings_seeds_and_deterministic_interleaving(self):
        self.assertEqual(stage.build_plan(self.source), self.plan)
        self.assertEqual((len(self.plan['arms']), len(self.plan['cells']), len(self.plan['jobs'])), (9, 180, 4554))
        counts = {}
        seeds = {}
        for job in self.plan['jobs']:
            role = job['metadata']['role']
            counts[role] = counts.get(role, 0) + 1
            if role != 'generation':
                self.assertNotIn('request', job)
                continue
            body, meta = job['request'], job['metadata']
            self.assertEqual({k: body[k] for k in self.generation if k not in ('seed', 'temperature')},
                             {k: v for k, v in self.generation.items() if k not in ('seed', 'temperature')})
            self.assertNotEqual(body['seed'], self.generation['seed'])
            key = (meta['cell']['task'], meta['cell']['repetition'], meta['index'])
            self.assertEqual(seeds.setdefault(key, body['seed']), body['seed'])
            self.assertNotIn('Unused instruction.', body['messages'][0]['content'])
        self.assertEqual(counts, {'generation': 1800, 'grouping': 414, 'quality': 2340})
        self.assertEqual(len(seeds), 200)
        self.assertEqual(self.plan['limits'], {'max_cost_usd': 20, 'max_requests': 9108, 'timeout_seconds': 600})
        for block in self.plan['schedule']['blocks']:
            self.assertEqual(len({j['arm'] for j in block['generation_order'][:9]}), 9)

    def test_named_observations_same_request_distinct_ids_canonical_labels(self):
        plan = self.compact()
        fake = FakeAPI(invalid_judge=True, alternate_groups=True)
        result = stage.execute(self.plan_path, self.output, fake)
        self.assertTrue(result['complete'])
        self.assertEqual(len(fake.calls), plan['initial_requests'] + 1)
        records = runtime.Journal(self.output, None, plan['limits'], ROOT).records()
        for cell in plan['cells']:
            groups = [r for r in records if r['metadata']['cell'] == cell
                      and r['metadata']['role'] == 'grouping' and r['metadata']['judge_role'] == 'primary']
            self.assertEqual(len(groups), 2)
            self.assertNotEqual(groups[0]['id'], groups[1]['id'])
            self.assertEqual(groups[0]['request'], groups[1]['request'])
            self.assertEqual(groups[0]['metadata']['order'], groups[1]['metadata']['order'])
            row = next(r for r in result['rows'] if r['cell'] == cell)
            self.assertNotEqual(*row['judges']['primary']['groupings'])
            for group in groups:
                self.assertEqual(group['attempts'][-1]['parsed'],
                    ev.parse_grouping(group['attempts'][-1]['text'], group['metadata']['order']))
                self.assertEqual(row['judges']['primary']['groupings'][group['metadata']['observation']],
                                 group['attempts'][-1]['parsed'])
        second = FakeAPI()
        self.assertEqual(stage.execute(self.plan_path, self.output, second), result)
        self.assertEqual(second.calls, [])

    def test_unknown_billing_stops_and_resume_reuses_saved_generation(self):
        plan = self.compact(retain_roster=True)
        first = FakeAPI(unknown_at=1)
        with self.assertRaisesRegex(ValueError, 'Unknown billing'):
            stage.execute(self.plan_path, self.output, first)
        result = json.loads((self.output / 'results.json').read_text())
        self.assertEqual(len(result['rows']), 180)
        self.assertFalse(result['complete'])
        self.assertFalse(result['accounting_complete'])
        second = FakeAPI()
        with self.assertRaisesRegex(ValueError, 'Unknown billing'):
            stage.execute(self.plan_path, self.output, second)
        self.assertEqual(second.calls, [])
        journal = runtime.Journal(self.output, None, plan['limits'], ROOT)
        journal.reconcile(plan['jobs'][0]['id'], 0, .001, 'Synthetic billing evidence')
        resumed = stage.execute(self.plan_path, self.output, second)
        self.assertTrue(resumed['complete'])
        self.assertEqual(len(second.calls), plan['initial_requests'] - 1)

    def test_invalid_observed_generation_is_not_regenerated(self):
        plan = self.compact()
        result = stage.execute(self.plan_path, self.output, FakeAPI(bad_generation=True))
        self.assertFalse(result['complete'])
        journal = runtime.Journal(self.output, None, plan['limits'], ROOT)
        bad = next(r for r in journal.records() if r['id'] == plan['jobs'][0]['id'])
        self.assertEqual(len(bad['attempts']), 1)
        self.assertIsNotNone(bad['attempts'][0]['response'])
        second = FakeAPI()
        stage.execute(self.plan_path, self.output, second)
        self.assertEqual(second.calls, [])

    def test_raw_response_and_dynamic_body_tampering_rejected_before_calls(self):
        plan = self.compact()
        stage.execute(self.plan_path, self.output, FakeAPI())
        generation = next(j for j in plan['jobs'] if j['metadata']['role'] == 'generation')
        path = self.output / 'requests' / (generation['id'] + '.json')
        saved = json.loads(path.read_text())
        changed = copy.deepcopy(saved)
        changed['attempts'][-1]['text'] += ' altered'
        op.write_json(path, changed)
        second = FakeAPI()
        with self.assertRaisesRegex(ValueError, 'raw response/text'):
            stage.execute(self.plan_path, self.output, second)
        self.assertEqual(second.calls, [])
        op.write_json(path, saved)
        judgment = next(j for j in plan['jobs'] if j['metadata']['role'] == 'grouping')
        path = self.output / 'requests' / (judgment['id'] + '.json')
        changed = json.loads(path.read_text())
        changed['request']['messages'][1]['content'] += ' altered'
        op.write_json(path, changed)
        with self.assertRaisesRegex(ValueError, 'frozen template'):
            stage.execute(self.plan_path, self.output, second)

    def test_unplanned_record_and_overlapping_output_rejected(self):
        plan = self.compact()
        with self.assertRaisesRegex(ValueError, 'outside the frozen source'):
            stage.execute(self.plan_path, self.source / 'new-run', FakeAPI())
        stage.execute(self.plan_path, self.output, FakeAPI())
        path = self.output / 'requests' / 'unplanned.json'
        op.write_json(path, {'id': 'unplanned', 'attempts': []})
        with self.assertRaisesRegex(ValueError, 'Unplanned request'):
            stage.execute(self.plan_path, self.output, FakeAPI())

    def test_plan_source_and_code_drift_rejected(self):
        self.plan['generation']['temperature'] = 1.5
        op.write_json(self.plan_path, self.plan)
        fake = FakeAPI()
        with self.assertRaisesRegex(ValueError, 'changed'):
            stage.execute(self.plan_path, self.output, fake)
        self.assertEqual(fake.calls, [])
        self.assertFalse(self.output.exists())
        (self.source / 'report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Frozen main report'):
            stage.build_plan(self.source)

    def test_live_guard_before_receipt_or_credentials_and_source_immutability(self):
        before = {p: data.sha(p) for p in self.source.glob('*.json')}
        with self.assertRaisesRegex(ValueError, 'explicit --approve-live'):
            stage.execute(self.plan_path, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {p: data.sha(p) for p in self.source.glob('*.json')})

    def test_budget_stop_has_no_attempt_and_preserves_original_error(self):
        plan = self.compact()
        plan['limits']['max_cost_usd'] = .001
        op.write_json(self.plan_path, plan)
        fake = FakeAPI()
        with self.assertRaisesRegex(ValueError, 'Cost budget would be exceeded'):
            stage.execute(self.plan_path, self.output, fake)
        self.assertEqual(fake.calls, [])
        result = json.loads((self.output / 'results.json').read_text())
        self.assertEqual(result['accounting']['attempted_requests'], 0)
        self.assertFalse(result['complete'])

    def test_final_unknown_billing_cannot_be_reported_complete(self):
        plan = self.compact()
        with self.assertRaisesRegex(ValueError, 'Unknown billing'):
            stage.execute(self.plan_path, self.output, FakeAPI(unknown_at=plan['initial_requests']))
        result = json.loads((self.output / 'results.json').read_text())
        self.assertFalse(result['complete'])
        self.assertEqual(sum(result['completed'].values()), plan['initial_requests'] - 1)

    def test_partial_offline_report_replays_raw_and_never_overwrites_evidence(self):
        with self.assertRaisesRegex(ValueError, 'Unknown billing'):
            stage.execute(self.plan_path, self.output, FakeAPI(unknown_at=1))
        before = {p: data.sha(p) for p in (self.output / 'requests').glob('*.json')}
        report = stage.write_report(self.plan_path, self.output)
        self.assertFalse(report['results']['complete'])
        self.assertEqual(len(report['results']['rows']), 180)
        self.assertEqual(before, {p: data.sha(p) for p in before})
        with self.assertRaisesRegex(ValueError, 'new file'):
            stage.write_report(self.plan_path, self.output)

    def test_embedding_loading_requires_offline_flags_and_isolated_network(self):
        with patch.dict(stage.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'HF_HUB_OFFLINE'):
                stage._offline_embedder()
        with patch.dict(stage.os.environ, {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}):
            with patch.object(stage.socket, 'if_nameindex', return_value=[(1, 'lo'), (2, 'eth0')]):
                with self.assertRaisesRegex(ValueError, 'network-isolated'):
                    stage._offline_embedder()


if __name__ == '__main__':
    unittest.main()
