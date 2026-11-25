"""Frozen, grouping-only sensitivity on the completed exploratory main run.

prepare and verify are offline. A live run additionally requires the exact
plan digest, an explicit approval flag, and a clean checkout at its committed
freeze. Original main files are inputs only. No generation or quality calls,
new human references, qualification decision, or prompt search occurs here.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time

import api_runner as op
import api_data as data
import api_evaluation as ev
import api_runtime as runtime
import runner_foundation_v1 as foundation
from api_env import api_key
from research_client import Client

HERE = Path(__file__).resolve().parent
SCHEMA = 'estr-saved-grouping-sensitivity-v1'
SOURCE_HASHES = {
    'manifest.json': 'fa54ce0ba7de92e261a883b82036d08dc002b6b5c52f3d9a4db77a7ac68dcb50',
    'results.json': '9989896b9d3ffd279e0db09dbbe7e63aef31b74b0e15e14ff00f1ba70a5747b9',
    'report.json': '883a541dd45868c1caa9d046564c30d4333d9d7e1ec623ad36f930285207715a',
}
TASKS = ['curated-0', 'curated-10', 'curated-16', 'curated-18', 'curated-2',
         'curated-6', 'curated-7', 'curated-8', 'narrative-dev-01', 'narrative-dev-02']
ARMS = ['default', 'naive_creative', 'manual_causal_shift_v1', 'manual_alternative_selection_v1']
LIMITS = {'max_requests': 240, 'max_cost_usd': 15, 'timeout_seconds': 600}
SEED = 20261115
FLAGS = {'development_only': True, 'exploratory': True, 'confirmatory': False,
         'main_eligible': False, 'qualified_at_twenty': False}


def _outside(path, protected, label):
    path, protected = Path(path).resolve(), Path(protected).resolve()
    op.require(path != protected and protected not in path.parents,
               f'{label} must be outside the frozen source run')


def _source(source_run):
    """Read only the explicitly pinned main artifacts and primary judgments."""
    source_run = Path(source_run).resolve()
    inputs = {}
    for name, digest in SOURCE_HASHES.items():
        path = source_run / name
        op.require(path.is_file() and data.sha(path) == digest, f'Frozen main {name} differs')
        inputs[str(path)] = digest
    manifest = json.loads((source_run / 'manifest.json').read_text())
    results = json.loads((source_run / 'results.json').read_text())
    spec = manifest['specification']
    cfg = spec['config']
    op.require(spec['phase'] == results['phase'] == 'main' and spec['n'] == results['n'] == 10,
               'Only the completed n=10 main source is supported')
    op.require(results['exploratory'] is True and results['confirmatory'] is False
               and results['main_eligible'] is False, 'Preserve the exploratory source status')
    op.require(results['accounting']['unknown_cost_attempts'] == 0, 'Source billing is unresolved')
    op.require(spec['selection']['grouping'] == 'gemini', 'Source primary differs')
    op.require(sorted(t['id'] for t in spec['tasks']) == sorted(TASKS), 'Source task roster differs')
    op.require([a['id'] for a in cfg['arms']] == ARMS and cfg['repetitions'] == 2,
               'Source arm/repetition scope differs')
    route = copy.deepcopy(next(j for j in cfg['judges'] if j['label'] == 'gemini'))
    op.require(route['model'] == 'google/gemini-4-hydrogen'
               and route['provider'] == 'google-ai-studio/flex' and route['json_schema'] is True
               and route['parameters'] == {'max_tokens': 4096, 'reasoning': {'effort': 'low'}},
               'Source Hydrogen parameters differ')
    snapshot = spec['snapshots']['j-gemini']
    snapshot_path = Path(snapshot['path']).resolve()
    op.require(data.sha(snapshot_path) == snapshot['sha256'], 'Saved Hydrogen endpoint snapshot differs')
    inputs[str(snapshot_path)] = snapshot['sha256']
    # The endpoint path is local provenance, never part of an API body.
    route['endpoint_snapshot'] = str(snapshot_path)
    endpoint = data.runtime_route(route, route['parameters'], HERE)
    tasks = {t['id']: t for t in spec['tasks']}
    cells = {}
    requests = {}
    for row in results['rows']:
        cell = row['cell']
        key = (cell['task'], cell['arm'], cell['repetition'])
        op.require(cell['generator'] == 'anchor' and key not in cells, 'Duplicate or wrong source cell')
        op.require(row['generation_complete'] and len(row['generations']) == 10,
                   'Source generations are incomplete')
        op.require(all(a['status'] == 'ok' and isinstance(a.get('text'), str)
                       for a in row['generations']), 'Unusable source generation')
        identity = op.fingerprint(cell)[:20]
        path = source_run / 'requests' / f'group-{identity}-primary.json'
        request = json.loads(path.read_text())
        inputs[str(path)] = data.sha(path)
        op.require(request['id'] == path.stem and request['metadata']['cell'] == cell
                   and request['metadata']['role'] == 'grouping', 'Source grouping identity differs')
        order = request['metadata']['order']
        expected = ev.judge_body(route, tasks[cell['task']]['text'],
            [a['text'] for a in row['generations']], 'grouping', order,
            profile_id=cfg['evaluation_profile'])
        op.require(request['request'] == expected, 'Source grouping body/response order differs')
        last = request['attempts'][-1]
        op.require(last['status'] == 'ok' and last.get('reported_cost_usd') is not None,
                   'Original primary grouping did not succeed with known billing')
        raw_text, finish = op.response_text(last['response'], generation=False)
        op.require(raw_text == last['text'] and finish == 'stop', 'Saved primary response/text differs')
        foundation.validate_returned_route(last['response'], route, endpoint)
        labels = ev.parse_grouping(last['text'], order)
        op.require(labels == last['parsed'] == row['judges']['primary']['labels'],
                   'Source grouping labels disagree with the saved response')
        cells[key], requests[key] = row, request
    expected_keys = {(t, a, r) for t in TASKS for a in ARMS for r in (0, 1)}
    op.require(set(cells) == expected_keys, 'Source must contain all 80 exact cells')
    return source_run, manifest, results, tasks, route, endpoint, inputs, cells, requests


def build_plan(source_run):
    """Deterministic full plan; does not create files, access a key, or call an API."""
    source_run, manifest, results, tasks, route, endpoint, inputs, rows, requests = _source(source_run)
    jobs, cells = [], []
    for task in TASKS:
        for arm in ARMS:
            original, pooled_ids, pooled_texts, parent_requests = [], [], [], []
            for repetition in (0, 1):
                row, saved = rows[task, arm, repetition], requests[task, arm, repetition]
                identity = op.fingerprint(row['cell'])[:20]
                response_ids = [f'g-{identity}-{i}' for i in range(10)]
                original.append({'repetition': repetition, 'response_ids': response_ids,
                                 'labels': row['judges']['primary']['labels']})
                pooled_ids.extend(response_ids)
                pooled_texts.extend(a['text'] for a in row['generations'])
                parent_requests.append(saved['id'])
                metadata = {'role': 'grouping', 'rejudge_kind': 'same_order_repeat',
                    'task_id': task, 'arm': arm, 'repetition': repetition,
                    'response_ids': response_ids, 'order': saved['metadata']['order'],
                    'source_request_ids': [saved['id']], 'evaluator': saved['metadata']['evaluator']}
                jobs.append({'id': f'repeat-{identity}', 'request': copy.deepcopy(saved['request']),
                             'metadata': metadata})
            # One fresh joint partition, with identities and repetition labels
            # retained only in local metadata. Index labels are anonymous 0..19.
            order = ev.shuffle(20, SEED, [SCHEMA, 'pooled', 'anchor', task, arm])
            metadata = {'role': 'grouping', 'rejudge_kind': 'pooled_twenty',
                'task_id': task, 'arm': arm, 'repetition': None, 'response_ids': pooled_ids,
                'order': order, 'source_request_ids': parent_requests,
                'evaluator': manifest['specification']['config']['evaluation_profile']}
            body = ev.judge_body(route, tasks[task]['text'], pooled_texts, 'grouping', order,
                                 profile_id=metadata['evaluator'])
            jobs.append({'id': 'pooled-' + op.fingerprint([task, arm])[:20],
                         'request': body, 'metadata': metadata})
            cells.append({'task_id': task, 'arm': arm, 'original': original,
                'repeats': [{'repetition': r['repetition'], 'response_ids': r['response_ids'],
                             'labels': None} for r in original],
                'pooled': {'response_ids': pooled_ids, 'labels': None}})
    # Random task blocks and within-block jobs keep measurement type and arm
    # from being confounded with a fixed time order. Original body order stays.
    task_order = TASKS.copy()
    random.Random(op.fingerprint([SEED, SCHEMA, 'tasks'])).shuffle(task_order)
    schedule = []
    for task in task_order:
        block = [job for job in jobs if job['metadata']['task_id'] == task]
        random.Random(op.fingerprint([SEED, SCHEMA, task])).shuffle(block)
        schedule.extend(block)
    reservations = {}
    for kind in ('same_order_repeat', 'pooled_twenty'):
        selected = [job for job in schedule if job['metadata']['rejudge_kind'] == kind]
        bounds = [foundation._reservation(job['request'], endpoint) for job in selected]
        op.require(all(upper + 4096 <= endpoint['context_length'] for upper, _ in bounds),
                   'A request exceeds the saved conservative context ceiling')
        amounts = [max(route['reserve_usd'], math.nextafter(float(cost), math.inf)) for _, cost in bounds]
        reservations[kind] = {'requests': len(selected), 'reserve_usd': sum(amounts),
            'maximum_request_reserve_usd': max(amounts),
            'minimum_input_token_upper_bound': min(x[0] for x in bounds),
            'maximum_input_token_upper_bound': max(x[0] for x in bounds)}
    initial = sum(v['reserve_usd'] for v in reservations.values())
    op.require(2 * initial <= LIMITS['max_cost_usd'], 'Full one-retry reserve exceeds the stage ceiling')
    return {'schema': SCHEMA, 'source_run': str(source_run), 'source_files_sha256': inputs,
        'source_accounting': results['accounting'], 'source_git_commit': manifest['git_commit'],
        'source_hashes': SOURCE_HASHES.copy(), 'code_sha256': runtime.code_hashes(),
        'route': route, 'snapshots': {'primary': {'path': route['endpoint_snapshot'],
            'sha256': endpoint['snapshot_sha256']}}, 'seed': SEED, 'limits': LIMITS.copy(),
        'monthly_key_boundary_usd': 50, 'minimum_request_start_interval_seconds': 2,
        'judge_retries': 1, 'initial_requests': 120, 'maximum_attempts': 240,
        'reservation': {'by_kind': reservations, 'initial_usd': initial,
                        'all_one_retry_usd': 2 * initial,
                        'saved_context_length': endpoint['context_length'],
                        'note': 'Saved endpoint metadata, not a fresh live verification or an expected bill.'},
        'jobs': schedule, 'cells': cells, **FLAGS}


def verify_plan(plan_path):
    path = Path(plan_path).resolve()
    saved = json.loads(path.read_text())
    op.require(saved.get('schema') == SCHEMA, 'Wrong saved sensitivity schema')
    _outside(path, saved['source_run'], 'Plan')
    expected = build_plan(saved['source_run'])
    op.require(saved == expected, 'Plan, implementation, or source lineage changed; prepare a new plan')
    return saved


def summary(plan, path=None):
    return {'schema': SCHEMA, 'initial_requests': plan['initial_requests'],
        'maximum_attempts': plan['maximum_attempts'], 'limits': plan['limits'],
        'reservation': plan['reservation'], 'approval': 'pending unless separately authorized',
        **({'plan_sha256': data.sha(path)} if path else {}), **FLAGS}


def verify_live_freeze(plan_path, approved_plan_sha256, freeze_commit, approve_live):
    op.require(approve_live is True, 'Live execution requires explicit --approve-live')
    op.require(re.fullmatch(r'[0-9a-f]{64}', approved_plan_sha256 or '') is not None
               and data.sha(plan_path) == approved_plan_sha256, 'Approved plan digest differs')
    op.require(re.fullmatch(r'[0-9a-f]{40}', freeze_commit or '') is not None,
               'Supply the full committed freeze ID')
    commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(HERE), 'status', '--porcelain'], text=True).strip()
    op.require(commit == freeze_commit and not dirty, 'Execution requires a clean checkout at the freeze commit')


class FrozenClient:
    """Credentials are deferred until validation and journal binding finish."""
    def __init__(self, plan, plan_path, receipt):
        self.plan, self.path = plan, Path(plan_path).resolve()
        self.digest, self.client, self.last_start = data.sha(self.path), None, None
        self.receipt_files = receipt['files_sha256']

    def verify_bindings(self):
        op.require(data.sha(self.path) == self.digest, 'Approved plan changed during execution')
        op.require(runtime.code_hashes() == self.plan['code_sha256'], 'Frozen implementation changed')
        op.require(all(data.sha(path) == digest for path, digest in self.plan['source_files_sha256'].items()),
                   'A frozen source input changed')
        op.require(all(data.sha(path) == digest for path, digest in self.receipt_files.items()),
                   'Preflight receipt evidence changed')

    def prepare(self):
        self.verify_bindings()
        self.client = Client(api_key(), self.plan['limits']['timeout_seconds'])

    def before_attempt(self):
        self.verify_bindings()

    def complete(self, body):
        op.require(self.client is not None, 'Client must be prepared after binding')
        if self.last_start is not None:
            remaining = 2 - (time.monotonic() - self.last_start)
            if remaining > 0:
                time.sleep(remaining)
        self.last_start = time.monotonic()
        return self.client.complete(body)


def collect_results(plan, journal):
    """A missing/invalid judgment stays missing; every attempted charge remains."""
    cells = copy.deepcopy(plan['cells'])
    lookup = {(c['task_id'], c['arm']): c for c in cells}
    records = {r['id']: r for r in journal.records()}
    op.require(len(records) == len(journal.records()), 'Duplicate saved request identities')
    jobs = {j['id']: j for j in plan['jobs']}
    op.require(set(records) <= set(jobs), 'Unplanned request in the stage journal')
    endpoint = data.runtime_route(plan['route'], plan['route']['parameters'], HERE)
    for identity, record in records.items():
        job = jobs[identity]
        op.require(record['request'] == job['request'] and record['metadata'] == job['metadata'],
                   'Saved journal request or metadata differs from the frozen job')
        op.require((journal.requests / (identity + '.json')).is_file(), 'Saved request filename differs')
        for attempt in record['attempts']:
            if attempt['status'] == 'ok':
                raw_text, finish = op.response_text(attempt['response'], generation=False)
                op.require(raw_text == attempt['text'] and finish == 'stop', 'Saved judge response/text differs')
                foundation.validate_returned_route(attempt['response'], plan['route'], endpoint)
                op.require(ev.parse_grouping(raw_text, job['metadata']['order']) == attempt['parsed'],
                           'Saved judge parse differs from its response and index map')
    completed = 0
    for job in plan['jobs']:
        record = records.get(job['id'])
        last = record['attempts'][-1] if record and record['attempts'] else None
        labels = (last.get('parsed') if last and last['status'] == 'ok'
                  and last.get('reported_cost_usd') is not None else None)
        meta = job['metadata']
        cell = lookup[meta['task_id'], meta['arm']]
        target = (cell['pooled'] if meta['rejudge_kind'] == 'pooled_twenty'
                  else cell['repeats'][meta['repetition']])
        target['labels'] = labels
        completed += labels is not None
    account = journal.account()
    attempts = [a for r in records.values() for a in r['attempts']]
    accounting_complete = (account['unknown_cost_attempts'] == 0
        and not any(a.get('routing_mismatch') or (a.get('setup_error') and not a.get('setup_resolution'))
                    or a.get('reported_cost_usd', 0) > a['reserve_usd'] for a in attempts))
    return {'schema': SCHEMA + '-results', 'plan_fingerprint': op.fingerprint(plan),
        'code_sha256': plan['code_sha256'], 'source_hashes': plan['source_hashes'],
        'source_run': plan['source_run'], 'source_accounting': plan['source_accounting'],
        'accounting': account, 'main_plus_stage_known_cost_usd':
            plan['source_accounting']['known_cost_usd'] + account['known_cost_usd'],
        'completed_judgments': completed, 'expected_judgments': 120,
        'judgments_complete': completed == 120, 'accounting_complete': accounting_complete,
        'complete': completed == 120 and accounting_complete, 'cells': cells, **FLAGS}


def execute(plan_path, output, client=None, *, freeze_commit=None,
            approved_plan_sha256=None, approve_live=False, preflight_receipt_dir=None):
    """Injected clients are for offline tests; CLI always uses the guarded client."""
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = verify_plan(plan_path)
    _outside(output, plan['source_run'], 'Output')
    op.require(plan_path != output and output not in plan_path.parents,
               'Plan must be outside the execution directory')
    op.require(all(output != Path(path) and output not in Path(path).parents
                   for path in plan['source_files_sha256']), 'Execution output overlaps frozen source inputs')
    if output.exists():
        op.require(output.is_dir(), 'Execution output must be a directory')
        op.require((output / 'manifest.json').is_file()
                   or not any(p.name != '.lock' for p in output.iterdir()),
                   'Existing output is not an empty or bound stage directory')
    live = client is None
    receipt = None
    if live:
        verify_live_freeze(plan_path, approved_plan_sha256, freeze_commit, approve_live)
        op.require(preflight_receipt_dir is not None, 'A fresh route and allowance receipt is required')
        from rejudge_preflight import verify_receipt
        receipt = verify_receipt(plan_path, preflight_receipt_dir)
        client = FrozenClient(plan, plan_path, receipt)
    specification = {'phase': SCHEMA, 'plan_sha256': data.sha(plan_path),
        'plan': plan, 'snapshots': plan['snapshots'], 'live': live,
        'freeze_commit': freeze_commit if live else None}
    with runtime.run_lock(output):
        runtime.bind_run(output, specification)
        journal = runtime.Journal(output, client, plan['limits'], HERE)
        previous = collect_results(plan, journal)
        if previous['complete']:
            return previous
        journal.check_budget(0)
        if live:
            (output / 'preflight_receipts').mkdir(exist_ok=True)
            op.write_json(output / 'preflight_receipts' / (op.fingerprint(receipt) + '.json'), receipt)
            client.prepare()
        try:
            for index, job in enumerate(plan['jobs']):
                order = job['metadata']['order']
                journal.call(job['id'], plan['route'], job['request'], job['metadata'],
                             parse=lambda text, order=order: ev.parse_grouping(text, order), retries=1)
                if live and (index == 0 or (index + 1) % 20 == 0):
                    print(json.dumps({'event': 'grouping_progress', 'scheduled_jobs': index + 1,
                                      'accounting': journal.account()}), flush=True)
        finally:
            result = collect_results(plan, journal)
            op.write_json(output / 'results.json', result)
            if live:
                client.verify_bindings()
        return result


def write_report(plan_path, output, destination=None):
    """Recompute a separate report from validated journal evidence, offline."""
    from rejudge_analysis import analyze
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = verify_plan(plan_path)
    _outside(output, plan['source_run'], 'Report source')
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    spec = manifest['specification']
    op.require(spec['phase'] == SCHEMA and spec['plan'] == plan
               and spec['plan_sha256'] == data.sha(plan_path)
               and manifest['code_sha256'] == plan['code_sha256'], 'Report run binding differs')
    destination = Path(destination).resolve() if destination else output / 'sensitivity-report.json'
    _outside(destination, plan['source_run'], 'Report destination')
    op.require(all(folder.resolve() != destination and folder.resolve() not in destination.parents
                   for folder in (output / 'requests', output / 'preflight_receipts')),
               'Report cannot be written into journal or preflight evidence directories')
    op.require(not destination.exists() and destination != plan_path
               and str(destination) not in plan['source_files_sha256'],
               'Report must use a new file outside source inputs and the plan')
    with runtime.run_lock(output):
        journal = runtime.Journal(output, None, plan['limits'], HERE)
        results = collect_results(plan, journal)
        report = {'schema': SCHEMA + '-report', 'plan_sha256': data.sha(plan_path),
            'manifest_sha256': data.sha(manifest_path),
            'request_files_sha256': {p.name: data.sha(p) for p in sorted(journal.requests.glob('*.json'))},
            'results': results, 'analysis': analyze(results['cells']), **FLAGS}
        destination.parent.mkdir(parents=True, exist_ok=True)
        op.write_json(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'report'))
    parser.add_argument('--source-run')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output')
    parser.add_argument('--freeze-commit')
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--approve-live', action='store_true')
    parser.add_argument('--preflight-receipt-dir')
    parser.add_argument('--report-json')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        op.require(args.source_run is not None, '--source-run is required')
        _outside(args.plan, args.source_run, 'Plan')
        op.require(not Path(args.plan).exists(), 'Preserve the existing plan; use verify or a new path')
        plan = build_plan(args.source_run)
        Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
        op.write_json(Path(args.plan), plan)
        print(json.dumps(summary(plan, args.plan), indent=2))
    elif args.command == 'verify':
        print(json.dumps(summary(verify_plan(args.plan), args.plan), indent=2))
    elif args.command == 'report':
        op.require(args.output is not None, '--output is required')
        report = write_report(args.plan, args.output, args.report_json)
        print(json.dumps({'report_created': True, 'complete': report['results']['complete'],
                          'accounting': report['results']['accounting']}))
    else:
        op.require(args.output is not None, '--output is required')
        result = execute(args.plan, args.output, freeze_commit=args.freeze_commit,
            approved_plan_sha256=args.approved_plan_sha256, approve_live=args.approve_live,
            preflight_receipt_dir=args.preflight_receipt_dir)
        print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
