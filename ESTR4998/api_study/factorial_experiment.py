"""Separate, prospective development-only temperature/instruction experiment.

Preparation, verification and journal replay are offline. Generation-dependent
judge requests are deterministic templates, instantiated only from the ten
validated saved outputs in canonical sample order. Two named primary grouping
observations use identical bodies; technical retries remain separate attempts.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import socket

import api_runner as op
import api_data as data
import api_evaluation as ev
import api_runtime as runtime
import runner_foundation_v1 as foundation
from rejudge_saved import (SOURCE_HASHES, TASKS, FrozenClient, _outside,
                           verify_live_freeze)

HERE = Path(__file__).resolve().parent
SCHEMA = 'estr-temperature-instruction-factorial-v1'
SEED = 20261116
TEMPERATURES = [0.8, 1.0, 1.2]
INSTRUCTIONS = ['default', 'naive_creative', 'manual_causal_shift_v1']
LIMITS = {'max_cost_usd': 20, 'max_requests': 9108, 'timeout_seconds': 600}
FLAGS = {'development_only': True, 'exploratory': True, 'confirmatory': False,
         'main_eligible': False, 'historical_failed_gates_preserved': True}


def _source(source_run):
    source_run = Path(source_run).resolve()
    inputs = {}
    for name, digest in SOURCE_HASHES.items():
        path = source_run / name
        op.require(path.is_file() and data.sha(path) == digest, f'Frozen main {name} differs')
        inputs[str(path)] = digest
    manifest = json.loads((source_run / 'manifest.json').read_text())
    results = json.loads((source_run / 'results.json').read_text())
    spec, cfg = manifest['specification'], manifest['specification']['config']
    op.require(spec['phase'] == results['phase'] == 'main' and spec['n'] == 10,
               'Only the pinned completed main experiment is supported')
    op.require(results['exploratory'] is True and results['confirmatory'] is False
               and results['main_eligible'] is False
               and results['accounting']['unknown_cost_attempts'] == 0,
               'Source status or accounting differs')
    op.require(sorted(t['id'] for t in spec['tasks']) == sorted(TASKS), 'Source task roster differs')
    op.require(cfg['repetitions'] == 2 and cfg['evaluation_profile'] == ev.TASK_SENSITIVE_PROFILE,
               'Source repetitions or evaluator differs')
    op.require(len(cfg['generators']) == 1 and cfg['generators'][0]['label'] == 'anchor',
               'Source generator differs')
    routes = {'generator': copy.deepcopy(cfg['generators'][0])}
    routes.update({role: copy.deepcopy(next(j for j in cfg['judges'] if j['label'] == label))
                   for role, label in (('primary', 'gemini'), ('check', 'gpt'))})
    snapshots = {}
    for role, source_key in [('generator', 'g-anchor-default'), ('primary', 'j-gemini'), ('check', 'j-gpt')]:
        entry = spec['snapshots'][source_key]
        path = Path(entry['path']).resolve()
        op.require(data.sha(path) == entry['sha256'], 'Frozen endpoint snapshot differs')
        inputs[str(path)] = entry['sha256']
        snapshots[role] = {'path': str(path), 'sha256': entry['sha256']}
        routes[role]['endpoint_snapshot'] = str(path)
        params = cfg['generation'] if role == 'generator' else routes[role]['parameters']
        data.runtime_route(routes[role], params, HERE)
    check_tasks = spec['check_tasks']['anchor']
    op.require(len(set(check_tasks)) == 3 and set(check_tasks) <= set(TASKS), 'Source check tasks differ')
    return source_run, manifest, results, inputs, routes, snapshots


def condition_id(instruction_id, temperature):
    return instruction_id + '-t' + format(temperature, '.1f').replace('.', 'p')


def _cell(block, job):
    return {'generator': block['generator'], 'task': block['task'],
            'arm': job['arm'], 'repetition': job['repetition']}


def _identity(cell):
    return op.fingerprint([SCHEMA, cell])[:20]


def build_plan(source_run):
    """Freeze every task, treatment, seed and request template without API access."""
    source_run, manifest, results, inputs, routes, snapshots = _source(source_run)
    spec, cfg = manifest['specification'], manifest['specification']['config']
    original = {arm['id']: arm for arm in cfg['arms']}
    op.require(all(not original[name]['decoding_overrides'] for name in INSTRUCTIONS),
               'Source instruction arm has an unexpected decoding override')
    arms = [{'id': condition_id(name, temp), 'instruction_id': name,
             'temperature': temp, 'instruction': original[name]['instruction']}
            for name in INSTRUCTIONS for temp in TEMPERATURES]
    generation = copy.deepcopy(cfg['generation']) | {'seed': SEED}
    tasks = copy.deepcopy(spec['tasks'])
    task_map, arm_map = {t['id']: t for t in tasks}, {a['id']: a for a in arms}
    check_tasks = copy.deepcopy(spec['check_tasks']['anchor'])
    schedule = foundation.schedule({'analysis': {'seed': SEED}, 'models': [routes['generator']],
        'arms': arms, 'repetitions': 2, 'samples_per_task': 10}, {'tasks': tasks})
    cells, jobs = [], []
    for block in schedule['blocks']:
        for job in block['generation_order']:
            cell = _cell(block, job)
            arm, task = arm_map[cell['arm']], task_map[cell['task']]
            # Common random-number seeds match sample positions across conditions;
            # they do not assert deterministic sampling or matched semantic outputs.
            seed = int(op.fingerprint([SEED, SCHEMA, task['id'], cell['repetition'], job['sample']])[:8], 16) & 0x7fffffff
            params = generation | {'temperature': arm['temperature'], 'seed': seed}
            text = task['text'] + ('\n\nAdditional instruction:\n' + arm['instruction'] if arm['instruction'] else '')
            jobs.append({'id': f'fg-{_identity(cell)}-{job["sample"]}', 'route': 'generator',
                'request': op.payload(routes['generator'], [{'role': 'user', 'content': text}], params),
                'metadata': {'role': 'generation', 'cell': cell, 'index': job['sample'], 'phase': SCHEMA}})
        measurements = []
        for job in block['measurement_order']:
            cell = _cell(block, job)
            cells.append(cell)
            identity = _identity(cell)
            response_ids = [f'fg-{identity}-{i}' for i in range(10)]
            order = ev.shuffle(10, SEED, [SCHEMA, identity])
            for role, count in [('primary', 2)] + ([('check', 1)] if cell['task'] in check_tasks else []):
                for observation in range(count):
                    measurements.append({'id': f'fg-group-{identity}-{role}-{observation}', 'route': role,
                        'response_ids': response_ids, 'metadata': {'role': 'grouping', 'cell': cell,
                            'judge_role': role, 'observation': observation, 'order': order,
                            'evaluator': cfg['evaluation_profile'], 'phase': SCHEMA}})
                for index in range(10):
                    measurements.append({'id': f'fg-score-{identity}-{role}-{index}', 'route': role,
                        'response_ids': [response_ids[index]], 'metadata': {'role': 'quality', 'cell': cell,
                            'judge_role': role, 'index': index, 'evaluator': cfg['evaluation_profile'], 'phase': SCHEMA}})
        # Separate repeated observations in random within-task positions, without
        # inspecting outputs or changing the shared anonymous response order.
        random.Random(op.fingerprint([SEED, SCHEMA, block['task'], 'measurements'])).shuffle(measurements)
        jobs.extend(measurements)
    op.require(len(cells) == 180 and len(jobs) == 4554, 'Unexpected factorial scope')
    endpoint = data.runtime_route(routes['generator'], generation, HERE)
    for job in jobs:
        if job['route'] == 'generator':
            upper, _ = foundation._reservation(job['request'], endpoint)
            op.require(upper + generation['max_tokens'] <= endpoint['context_length'], 'Generation context exceeds limit')
    from factorial_analysis import analysis_protocol
    return {'schema': SCHEMA, 'source_run': str(source_run), 'source_files_sha256': inputs,
        'source_hashes': SOURCE_HASHES.copy(), 'source_accounting': results['accounting'],
        'source_git_commit': manifest['git_commit'], 'code_sha256': runtime.code_hashes(),
        'routes': routes, 'snapshots': snapshots, 'generation': generation,
        'evaluation_profile': cfg['evaluation_profile'], 'tasks': tasks, 'arms': arms,
        'check_tasks': check_tasks, 'seed': SEED, 'n': 10, 'repetitions': 2,
        'limits': LIMITS.copy(), 'monthly_key_boundary_usd': 50,
        'minimum_request_start_interval_seconds': 2, 'judge_retries': 1,
        'initial_requests': 4554, 'maximum_attempts': 9108, 'schedule': schedule,
        'jobs': jobs, 'cells': cells, 'reference_pool': None,
        'reservation_policy': 'Each concrete request reserves its conservative saved-endpoint price or declared route floor, whichever is larger. The stage cap is enforced against committed spend plus the next reserve; completion within the cap is not guaranteed.',
        'analysis': analysis_protocol(arms), **FLAGS}


def verify_plan(plan_path):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    op.require(plan.get('schema') == SCHEMA, 'Wrong factorial plan schema')
    _outside(path, plan['source_run'], 'Plan')
    op.require(plan == build_plan(plan['source_run']), 'Plan, implementation or frozen source changed')
    return plan


def materialize_job(plan, job, records):
    """Bind exact judge bodies to validated outputs; incomplete inputs yield no job."""
    if job['metadata']['role'] == 'generation':
        return job['request'], job['metadata']
    attempts = []
    for identity in job['response_ids']:
        record = records.get(identity)
        last = record['attempts'][-1] if record and record['attempts'] else None
        if last is None or last['status'] != 'ok' or last.get('reported_cost_usd') is None:
            return None
        attempts.append(last)
    texts = [a['text'] for a in attempts]
    meta = copy.deepcopy(job['metadata'])
    task = next(t for t in plan['tasks'] if t['id'] == meta['cell']['task'])
    body = ev.judge_body(plan['routes'][job['route']], task['text'], texts, meta['role'],
                         meta.get('order'), profile_id=plan['evaluation_profile'])
    # These hashes are local lineage metadata, never provider-visible strings.
    meta['response_ids'] = job['response_ids']
    meta['response_text_sha256'] = [op.hashlib.sha256(text.encode()).hexdigest() for text in texts]
    return body, meta


def _validate_records(plan, journal):
    # Journal.call allocates an in-memory record before checking the next
    # reservation/binding. A stopped pre-submission allocation is not an attempt.
    saved = [r for r in journal.records() if r['attempts']
             or (journal.requests / (r['id'] + '.json')).exists()]
    records = {r['id']: r for r in saved}
    op.require(len(records) == len(saved), 'Duplicate saved request identity')
    jobs = {j['id']: j for j in plan['jobs']}
    op.require(set(records) <= set(jobs), 'Unplanned request in factorial journal')
    # Generation records precede judgments in this validation pass so no judged
    # input can silently rely on an edited raw response or stale derived text.
    ordered = sorted(records, key=lambda identity: jobs[identity]['route'] != 'generator')
    endpoints = {role: data.runtime_route(route, plan['generation'] if role == 'generator' else route['parameters'], HERE)
                 for role, route in plan['routes'].items()}
    for identity in ordered:
        record, job = records[identity], jobs[identity]
        expected = materialize_job(plan, job, records)
        op.require(expected is not None, 'Judgment exists without complete bound generation inputs')
        body, metadata = expected
        op.require(record['request'] == body and record['metadata'] == metadata,
                   'Saved request or metadata differs from the frozen template and inputs')
        op.require((journal.requests / (identity + '.json')).is_file(), 'Saved request filename differs')
        attempts = record['attempts']
        op.require(0 < len(attempts) <= 2, 'Unexpected request attempt count')
        for index, attempt in enumerate(attempts):
            op.require(attempt['attempt'] == index, 'Attempt index differs')
            if index and metadata['role'] == 'generation':
                op.require(runtime.no_reply_retryable(attempts[index - 1]), 'Observed generation was regenerated')
            if attempt['status'] == 'ok':
                text, finish = op.response_text(attempt['response'], generation=metadata['role'] == 'generation')
                op.require(text == attempt['text'] and finish == attempt['finish_reason'], 'Saved raw response/text differs')
                foundation.validate_returned_route(attempt['response'], plan['routes'][job['route']], endpoints[job['route']])
                if metadata['role'] != 'generation':
                    op.require(finish == 'stop', 'Truncated judge response')
                    parsed = (ev.parse_grouping(text, metadata['order']) if metadata['role'] == 'grouping'
                              else ev.parse_quality(text))
                    op.require(parsed == attempt['parsed'], 'Saved parse differs from raw response')
    return records


def collect_results(plan, journal):
    """Replay validated raw records; absent/failed observations remain explicit."""
    records = _validate_records(plan, journal)
    arm_map = {a['id']: a for a in plan['arms']}
    rows, row_map = [], {}
    for cell in plan['cells']:
        arm = arm_map[cell['arm']]
        row = {'cell': copy.deepcopy(cell), 'condition': {'instruction_id': arm['instruction_id'],
            'temperature': arm['temperature']}, 'samples': 10,
            'generations': [None] * 10, 'judges': {'primary': {'groupings': [None, None], 'scores': [None] * 10}}}
        if cell['task'] in plan['check_tasks']:
            row['judges']['check'] = {'groupings': [None], 'scores': [None] * 10}
        rows.append(row)
        row_map[op.fingerprint(cell)] = row
    completed = {'generation': 0, 'grouping': 0, 'quality': 0}
    for job in plan['jobs']:
        record = records.get(job['id'])
        last = record['attempts'][-1] if record and record['attempts'] else None
        meta = job['metadata']
        row = row_map[op.fingerprint(meta['cell'])]
        success = last is not None and last['status'] == 'ok' and last.get('reported_cost_usd') is not None
        completed[meta['role']] += bool(success)
        if meta['role'] == 'generation':
            row['generations'][meta['index']] = last
        elif success:
            target = row['judges'][meta['judge_role']]
            target['groupings' if meta['role'] == 'grouping' else 'scores'][
                meta['observation'] if meta['role'] == 'grouping' else meta['index']] = last['parsed']
    for row in rows:
        row['generation_complete'] = all(a and a['status'] == 'ok' and a.get('reported_cost_usd') is not None
                                         for a in row['generations'])
        for target in row['judges'].values():
            target['grouping_complete'] = all(a is not None for a in target['groupings'])
            target['quality_complete'] = all(a is not None for a in target['scores'])
    accounting = journal.account()
    attempts = [a for record in records.values() for a in record['attempts']]
    accounting_complete = (accounting['unknown_cost_attempts'] == 0
        and not any(a.get('routing_mismatch') or (a.get('setup_error') and not a.get('setup_resolution'))
                    or a.get('reported_cost_usd', 0) > a['reserve_usd'] for a in attempts))
    observations_complete = sum(completed.values()) == plan['initial_requests']
    return {'schema': SCHEMA + '-results', 'phase': SCHEMA, 'n': 10,
        'plan_fingerprint': op.fingerprint(plan), 'code_sha256': plan['code_sha256'],
        'completed': completed, 'initial_requests': plan['initial_requests'],
        'accounting': accounting, 'accounting_complete': accounting_complete,
        'observations_complete': observations_complete,
        'complete': observations_complete and accounting_complete, 'rows': rows, **FLAGS}


def _safe_output(plan, plan_path, output):
    _outside(output, plan['source_run'], 'Output')
    op.require(plan_path != output and output not in plan_path.parents,
               'Plan must be outside the execution directory')
    op.require(all(output != Path(path) and output not in Path(path).parents
                   for path in plan['source_files_sha256']), 'Output overlaps source inputs')
    if output.exists():
        op.require(output.is_dir() and ((output / 'manifest.json').is_file()
                   or not any(p.name != '.lock' for p in output.iterdir())), 'Output must be empty or a bound stage')


def execute(plan_path, output, client=None, *, freeze_commit=None, approved_plan_sha256=None,
            approve_live=False, preflight_receipt_dir=None):
    """CLI live launches require approval/freeze/receipt; injection is for tests."""
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = verify_plan(plan_path)
    _safe_output(plan, plan_path, output)
    live, receipt = client is None, None
    if live:
        verify_live_freeze(plan_path, approved_plan_sha256, freeze_commit, approve_live)
        op.require(preflight_receipt_dir is not None, 'Fresh route and allowance receipt required')
        from factorial_preflight import verify_receipt
        receipt = verify_receipt(plan_path, preflight_receipt_dir, expected_freeze_commit=freeze_commit)
        client = FrozenClient(plan, plan_path, receipt)
    specification = {'phase': SCHEMA, 'plan_sha256': data.sha(plan_path), 'plan': plan,
        'snapshots': plan['snapshots'], 'live': live, 'freeze_commit': freeze_commit if live else None}
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
                concrete = materialize_job(plan, job, {r['id']: r for r in journal.records()})
                if concrete is None:
                    continue
                body, metadata = concrete
                parse = None
                if metadata['role'] == 'grouping':
                    parse = lambda text, order=metadata['order']: ev.parse_grouping(text, order)
                elif metadata['role'] == 'quality':
                    parse = ev.parse_quality
                journal.call(job['id'], plan['routes'][job['route']], body, metadata,
                             parse=parse, retries=plan['judge_retries'])
                # Per-call requests are durable already; compact checkpoints every
                # twenty schedule positions make partial support visible as well.
                if (index + 1) % 20 == 0:
                    op.write_json(output / 'results.json', collect_results(plan, journal))
                    if live:
                        print(json.dumps({'event': 'factorial_progress', 'scheduled_jobs': index + 1,
                                          'accounting': journal.account()}), flush=True)
        finally:
            result = collect_results(plan, journal)
            op.write_json(output / 'results.json', result)
            if live:
                client.verify_bindings()
        return result


def _offline_embedder():
    """Load the pinned cache only inside an externally isolated network namespace."""
    op.require(os.environ.get('HF_HUB_OFFLINE') == '1'
               and os.environ.get('TRANSFORMERS_OFFLINE') == '1',
               'Embedding reports require HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1')
    op.require({name for _, name in socket.if_nameindex()} <= {'lo'},
               'Embedding reports require a network-isolated namespace (unshare -rn)')
    _, metrics = data.local_modules()
    return metrics.Embedder(data.EMBEDDING['model_id'], data.EMBEDDING['revision'],
                            device='cpu', window=data.EMBEDDING['window'])


def write_report(plan_path, output, destination=None, *, embeddings=False):
    """Offline replay into a new report, preserving raw evidence and old reports."""
    from factorial_analysis import analyze, with_embedding_metrics
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = verify_plan(plan_path)
    _safe_output(plan, plan_path, output)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    spec = manifest['specification']
    op.require(spec['phase'] == SCHEMA and spec['plan'] == plan
               and spec['plan_sha256'] == data.sha(plan_path)
               and manifest['code_sha256'] == plan['code_sha256'], 'Report run binding differs')
    destination = Path(destination).resolve() if destination else output / 'factorial-report.json'
    _outside(destination, plan['source_run'], 'Report destination')
    op.require(all(folder.resolve() != destination and folder.resolve() not in destination.parents
                   for folder in (output / 'requests', output / 'preflight_receipts')),
               'Report cannot overwrite journal or preflight evidence')
    op.require(not destination.exists() and destination != plan_path
               and str(destination) not in plan['source_files_sha256'], 'Report requires a new file')
    with runtime.run_lock(output):
        journal = runtime.Journal(output, None, plan['limits'], HERE)
        results = collect_results(plan, journal)
        if embeddings:
            results = with_embedding_metrics(results, _offline_embedder())
        report = {'schema': SCHEMA + '-report', 'plan_sha256': data.sha(plan_path),
            'manifest_sha256': data.sha(manifest_path),
            'request_files_sha256': {p.name: data.sha(p) for p in sorted(journal.requests.glob('*.json'))},
            'embedding_execution': {'requested': embeddings, 'network_isolated': embeddings,
                                    'cache_only': embeddings},
            'results': results, 'analysis': analyze(plan, results), **FLAGS}
        destination.parent.mkdir(parents=True, exist_ok=True)
        op.write_json(destination, report)
    return report


def summary(plan, path=None):
    return {'schema': SCHEMA, 'initial_requests': plan['initial_requests'],
        'maximum_attempts': plan['maximum_attempts'], 'limits': plan['limits'],
        'tasks': len(plan['tasks']), 'conditions': len(plan['arms']),
        'reservation_policy': plan['reservation_policy'],
        'approval': 'pending unless separately authorized',
        **({'plan_sha256': data.sha(path)} if path else {}), **FLAGS}


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
    parser.add_argument('--embeddings', action='store_true',
                        help='Report canonical cached MiniLM metrics with offline flags under unshare -rn')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        op.require(args.source_run is not None, '--source-run required')
        _outside(args.plan, args.source_run, 'Plan')
        op.require(not Path(args.plan).exists(), 'Preserve existing plan; use a new path')
        plan = build_plan(args.source_run)
        Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
        op.write_json(Path(args.plan), plan)
        print(json.dumps(summary(plan, args.plan), indent=2))
    elif args.command == 'verify':
        print(json.dumps(summary(verify_plan(args.plan), args.plan), indent=2))
    elif args.command == 'report':
        op.require(args.output is not None, '--output required')
        report = write_report(args.plan, args.output, args.report_json, embeddings=args.embeddings)
        print(json.dumps({'report_created': True, 'complete': report['results']['complete']}))
    else:
        op.require(args.output is not None, '--output required')
        result = execute(args.plan, args.output, freeze_commit=args.freeze_commit,
            approved_plan_sha256=args.approved_plan_sha256, approve_live=args.approve_live,
            preflight_receipt_dir=args.preflight_receipt_dir)
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
