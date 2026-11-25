"""Frozen second-generator technical smoke and causal-instruction adherence gate.

This is a prospective usability check, not diversity selection or qualification.
All preparation/replay functions are offline. Only an explicitly approved run
with a clean freeze and fresh preflight may construct a credentialed client.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import api_runner as op
import api_data as data
import api_evaluation as ev
import api_runtime as runtime
import runner_foundation_v1 as foundation
from factorial_analysis import response_diagnostics
from api_screen import parse_adherence
from rejudge_saved import FrozenClient, _outside, verify_live_freeze

HERE = Path(__file__).resolve().parent
SCHEMA = 'estr-second-generator-readiness-v1'
SEED = 20261117
SMOKES = (('curated-16', .8), ('curated-0', 1.0), ('narrative-dev-02', 1.2))
SCREEN_TASKS = ('curated-0', 'narrative-dev-02')
CAUSAL = 'manual_causal_shift_v1'
ADHERENCE_SCHEMA = {'type': 'object', 'properties': {
    'pass': {'type': 'boolean'}, 'reason': {'type': 'string'}},
    'required': ['pass', 'reason'], 'additionalProperties': False}


def _original_screen(candidate):
    """Recover the original adopted rubric from its already-bound main source."""
    manifest = json.loads((Path(candidate['source_run']) / 'manifest.json').read_text())
    authorization = manifest['specification']['exploratory_authorization']
    digest = authorization['adoption']['screen_spec']['sha256']
    paths = [Path(p).resolve() for p, value in authorization['inputs_sha256'].items()
             if value == digest and Path(p).name == authorization['adoption']['screen_spec']['path']]
    op.require(len(paths) == 1, 'Original adopted screen specification changed')
    path = paths[0]
    if not path.exists():
        # An ignored approved config may not have been copied into the historical
        # worktree. Only its deterministic source API/configs location is eligible;
        # identical frozen bytes are required and the actual path is newly bound.
        path = (Path(candidate['source_run']).resolve().parent.parent / 'configs' / path.name).resolve()
    op.require(path.is_file() and data.sha(path) == digest, 'Original adopted screen specification changed')
    spec = json.loads(path.read_text())
    op.require(spec['task_ids'] == list(SCREEN_TASKS) and spec['samples_per_task'] == 5
               and spec['minimum_passes_per_task'] == 4 and CAUSAL in spec['arm_ids']
               and isinstance(spec['judge_prompt'], str) and spec['judge_prompt'].strip(),
               'Original screen tasks, size, gate or rubric differ')
    return spec, {str(path): digest}


def build_readiness_plan(candidate_plan_path):
    """Freeze all 23 requests before any smoke or screen outcome is available."""
    from second_generator_experiment import verify_candidate_plan
    path = Path(candidate_plan_path).resolve()
    candidate = verify_candidate_plan(path)
    spec, screen_files = _original_screen(candidate)
    tasks = {task['id']: task for task in candidate['tasks']}
    causal = next(a for a in candidate['arms'] if a['instruction_id'] == CAUSAL and a['temperature'] == .8)
    default = next(a for a in candidate['arms'] if a['instruction_id'] == 'default' and a['temperature'] == .8)
    op.require(not default['instruction'] and candidate['seed'] == SEED
               and candidate['generation']['reasoning'] == {'effort': 'none'},
               'Readiness requires the frozen default and explicit nonreasoning controls')
    jobs = []

    def generation(phase, task_id, index, temperature, instruction):
        identity = op.fingerprint([SCHEMA, phase, task_id, index])[:20]
        params = candidate['generation'] | {'temperature': temperature, 'seed':
            int(op.fingerprint([SEED, SCHEMA, phase, task_id, index])[:8], 16) & 0x7fffffff}
        text = tasks[task_id]['text'] + ('\n\nAdditional instruction:\n' + instruction if instruction else '')
        job = {'id': 'sg-ready-g-' + identity, 'route': 'generator',
            'request': op.payload(candidate['routes']['generator'], [{'role': 'user', 'content': text}], params),
            'metadata': {'role': 'generation', 'phase': SCHEMA, 'stage': phase,
                'task': task_id, 'index': index, 'instruction_id': 'default' if phase == 'smoke' else CAUSAL}}
        jobs.append(job)
        return job

    for index, (task_id, temperature) in enumerate(SMOKES):
        generation('smoke', task_id, index, temperature, '')
    for task_id in SCREEN_TASKS:
        for index in range(5):
            generated = generation('screen', task_id, index, .8, causal['instruction'])
            jobs.append({'id': generated['id'].replace('-g-', '-j-'), 'route': 'primary',
                'response_id': generated['id'], 'metadata': {
                    'role': 'task_adherence', 'phase': SCHEMA, 'stage': 'screen',
                    'task': task_id, 'index': index, 'instruction_id': CAUSAL}})
    inputs = candidate['source_files_sha256'] | screen_files | {str(path): data.sha(path)}
    return {'schema': SCHEMA, 'candidate_plan_path': str(path), 'candidate_plan_sha256': data.sha(path),
        'candidate_fingerprint': op.fingerprint(candidate), 'source_run': candidate['source_run'],
        'source_files_sha256': inputs, 'code_sha256': runtime.code_hashes(),
        'routes': copy.deepcopy(candidate['routes']), 'snapshots': copy.deepcopy(candidate['snapshots']),
        'generation': copy.deepcopy(candidate['generation']), 'tasks': copy.deepcopy(candidate['tasks']),
        'seed': SEED, 'smokes': [{'task': task, 'temperature': temp, 'instruction_id': 'default'}
                               for task, temp in SMOKES],
        'screen': {'task_ids': list(SCREEN_TASKS), 'instruction_id': CAUSAL,
            'instruction': causal['instruction'], 'temperature': .8, 'samples_per_task': 5,
            'minimum_passes_per_task': 4, 'judge_prompt': spec['judge_prompt'],
            'original_spec_sha256': next(iter(screen_files.values())),
            'constraint_gate': 'v0.2/paired-emphasis-sentence-sensitivity',
            'constraint_diagnostic': 'frozen_original_parser',
            'pass_rule': 'binary adherence pass AND applicable v0.2 constraints pass AND nonempty AND nonrefusal'},
        'technical_gate': 'Every generation: valid exact route, known billing, status ok, nonempty final text, finish_reason stop, explicitly reported reasoning_tokens integer zero.',
        'failure_policy': 'Stop on technical failure; never regenerate observed output. One eligible known-cost no-reply generation retry or one technical judgment retry. A valid binary false is retained without retry. No tuning or candidate replacement within this plan.',
        'limits': {'max_cost_usd': 20, 'max_requests': 46, 'timeout_seconds': 600},
        'combined_limits': copy.deepcopy(candidate['combined_limits']), 'monthly_key_boundary_usd': 50,
        'initial_requests': 23, 'maximum_attempts': 46, 'judge_retries': 1,
        'minimum_request_start_interval_seconds': 2, 'jobs': jobs,
        'note': 'Preparation costs and attempts must be deducted from the combined main-stage allowance. This gate does not measure diversity or establish evaluator qualification.'}


def verify_plan(plan_path):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    op.require(plan.get('schema') == SCHEMA, 'Wrong readiness schema')
    _outside(path, plan['source_run'], 'Readiness plan')
    op.require(plan == build_readiness_plan(plan['candidate_plan_path']),
               'Readiness plan, code, candidate or source changed')
    return plan


def _technical_errors(attempt):
    if not attempt:
        return ['missing_generation']
    errors = []
    if attempt.get('status') != 'ok':
        errors.append('unsuccessful_generation')
    if attempt.get('reported_cost_usd') is None:
        errors.append('unknown_billing')
    if not isinstance(attempt.get('text'), str) or not attempt['text'].strip():
        errors.append('empty_or_missing_final_text')
    if attempt.get('finish_reason') != 'stop':
        errors.append('nonstop_finish_reason')
    if op.reasoning_tokens(attempt.get('response')) != 0:
        errors.append('reasoning_not_explicitly_zero')
    raw = attempt.get('response')
    choices = raw.get('choices', []) if isinstance(raw, dict) else []
    choices = choices if isinstance(choices, list) else []
    if any(isinstance(choice, dict) and isinstance(choice.get('message'), dict) and
           any(choice['message'].get(field) for field in ('reasoning', 'reasoning_content', 'reasoning_details'))
           for choice in choices):
        errors.append('observed_reasoning_content')
    if attempt.get('routing_mismatch'):
        errors.append('routing_mismatch')
    return errors


def _last(records, identity):
    attempts = records.get(identity, {}).get('attempts', [])
    return attempts[-1] if attempts else None


def materialize_job(plan, job, records):
    if job['route'] == 'generator':
        return job['request'], job['metadata']
    generation = _last(records, job['response_id'])
    if _technical_errors(generation):
        return None
    task = next(t for t in plan['tasks'] if t['id'] == job['metadata']['task'])
    judge = plan['routes']['primary']
    body = op.payload(judge, [{'role': 'system', 'content': plan['screen']['judge_prompt']},
        {'role': 'user', 'content': ev.messages(task['text'], [generation['text']], 'quality')[1]['content']}],
        judge['parameters'])
    if judge['json_schema']:
        body['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'task_adherence', 'strict': True, 'schema': ADHERENCE_SCHEMA}}
    else:
        body['messages'][0]['content'] += '\nReturn JSON only: {"pass":true,"reason":"..."}.'
    return body, job['metadata'] | {'response_id': job['response_id'],
        'response_text_sha256': op.hashlib.sha256(generation['text'].encode()).hexdigest()}


def _validate_records(plan, journal):
    saved = [r for r in journal.records() if r['attempts'] or
             (journal.requests / (r['id'] + '.json')).exists()]
    records = {r['id']: r for r in saved}
    jobs = {j['id']: j for j in plan['jobs']}
    op.require(len(records) == len(saved) and set(records) <= set(jobs), 'Unplanned readiness request')
    for job in plan['jobs']:
        record = records.get(job['id'])
        if not record:
            continue
        concrete = materialize_job(plan, job, records)
        op.require(concrete is not None, 'Adherence judgment without technically passing generation')
        body, metadata = concrete
        op.require(record['request'] == body and record['metadata'] == metadata
                   and (journal.requests / (job['id'] + '.json')).is_file(),
                   'Saved readiness request or lineage differs')
        attempts = record['attempts']
        op.require(0 < len(attempts) <= 2, 'Unexpected readiness attempt count')
        endpoint = data.runtime_route(plan['routes'][job['route']],
            {k: v for k, v in body.items() if k in op.PARAMETERS}, HERE)
        input_upper, priced_reserve = foundation._reservation(body, endpoint)
        reserve = max(plan['routes'][job['route']]['reserve_usd'],
                      math.nextafter(float(priced_reserve), math.inf))
        for index, attempt in enumerate(attempts):
            op.require(attempt['attempt'] == index, 'Attempt index differs')
            if index:
                if metadata['role'] == 'generation':
                    op.require(runtime.no_reply_retryable(attempts[index - 1]), 'Observed generation was regenerated')
                else:
                    op.require(attempts[index - 1]['status'] != 'ok', 'Valid adherence outcome was retried')
            cost = attempt.get('reported_cost_usd')
            op.require(cost is None or (type(cost) in (int, float) and math.isfinite(cost) and cost >= 0),
                       'Invalid saved accounting')
            op.require(attempt['snapshot_sha256'] == endpoint['snapshot_sha256']
                       and attempt['reserve_usd'] == reserve
                       and attempt['priced_reserve_decimal_usd'] == str(priced_reserve)
                       and attempt['input_token_upper_bound'] == input_upper
                       and attempt['pricing_per_token'] == endpoint['pricing'],
                       'Attempt snapshot or reservation differs')
            raw = attempt.get('response')
            usage = raw.get('usage') if isinstance(raw, dict) else None
            reported = usage.get('cost') if isinstance(usage, dict) else None
            history = attempt.get('billing_history', [])
            if history:
                op.require(history[-1]['cost'] == cost and all(isinstance(h.get('note'), str)
                    and h['note'].strip() and h.get('utc') for h in history), 'Malformed billing reconciliation history')
            if type(reported) in (int, float) and math.isfinite(reported) and reported >= 0:
                op.require(cost == reported or bool(history), 'Saved billing differs without reconciliation history')
            if attempt['status'] == 'ok':
                raw = attempt['response']
                text, finish = op.response_text(raw, generation=metadata['role'] == 'generation')
                op.require(text == attempt['text'] and finish == attempt['finish_reason']
                           and op.reasoning_tokens(raw) == attempt.get('reasoning_tokens'), 'Saved raw response differs')
                foundation.validate_returned_route(raw, plan['routes'][job['route']], endpoint)
                if metadata['role'] == 'task_adherence':
                    op.require(finish == 'stop' and parse_adherence(text) == attempt['parsed'],
                               'Saved adherence parse differs')
    # No downstream submission is allowed after an incomplete/technical failure.
    blocked = False
    for job in plan['jobs']:
        attempt = _last(records, job['id'])
        op.require(not (blocked and attempt is not None), 'Readiness continued after an incomplete or failed prerequisite')
        ready = not _technical_errors(attempt) if job['route'] == 'generator' else bool(
            attempt and attempt['status'] == 'ok' and attempt.get('reported_cost_usd') is not None)
        blocked = blocked or not ready
    return records


def collect_results(plan, journal):
    """Count conjunction passes per task; five responses give a fixed denominator.

    A missing judgment is missing support, not a favorable result. Both task
    blocks must independently reach four of five complete retained outcomes.
    The old deterministic parser is a diagnostic and cannot override v0.2.
    """
    records = _validate_records(plan, journal)
    smokes, outcomes = [], []
    completed = 0
    for job in plan['jobs']:
        attempt = _last(records, job['id'])
        complete = bool(attempt and attempt['status'] == 'ok' and attempt.get('reported_cost_usd') is not None)
        completed += complete
        if job['route'] == 'generator' and job['metadata']['stage'] == 'smoke':
            smokes.append({'request_id': job['id'], 'task': job['metadata']['task'],
                'temperature': job['request']['temperature'], 'passed': not _technical_errors(attempt),
                'technical_errors': _technical_errors(attempt)})
        elif job['route'] == 'primary':
            generation = _last(records, job['response_id'])
            diagnostic = response_diagnostics(job['metadata']['task'], generation)
            ready = not _technical_errors(generation)
            binary = attempt['parsed']['pass'] if complete else None
            passed = bool(ready and binary is True and diagnostic['constraint_satisfied'] is not False
                          and not diagnostic['flags']['empty'] and not diagnostic['flags']['refusal'])
            outcomes.append({'task': job['metadata']['task'], 'index': job['metadata']['index'],
                'generation_id': job['response_id'], 'adherence_id': job['id'],
                'complete': ready and complete, 'technical_errors': _technical_errors(generation),
                'binary_adherence': binary, 'diagnostics': diagnostic, 'passed': passed})
    blocks = [{'task': task, 'completed': sum(r['complete'] for r in outcomes if r['task'] == task),
        'passes': sum(r['passed'] for r in outcomes if r['task'] == task), 'required': 4, 'denominator': 5}
        for task in SCREEN_TASKS]
    for block in blocks:
        block['passed'] = block['completed'] == 5 and block['passes'] >= 4
    account = journal.account()
    attempts = [a for record in records.values() for a in record['attempts']]
    accounting_complete = (account['unknown_cost_attempts'] == 0
        and account['attempted_requests'] <= plan['limits']['max_requests']
        and account['committed_usd'] <= plan['limits']['max_cost_usd'] and not any(
        a.get('routing_mismatch') or (a.get('setup_error') and not a.get('setup_resolution'))
        or a.get('reported_cost_usd', 0) > a['reserve_usd'] for a in attempts))
    complete = completed == 23 and accounting_complete and all(s['passed'] for s in smokes)
    return {'schema': SCHEMA + '-results', 'plan_fingerprint': op.fingerprint(plan),
        'candidate_fingerprint': plan['candidate_fingerprint'], 'code_sha256': plan['code_sha256'],
        'complete': complete, 'passed': complete and all(b['passed'] for b in blocks),
        'completed_requests': completed, 'accounting': account, 'smokes': smokes,
        'blocks': blocks, 'outcomes': outcomes}


def _safe_output(plan, plan_path, output):
    from factorial_experiment import _safe_output as check_output
    check_output(plan, plan_path, output)


def _receipt(plan, plan_path, output, result):
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    spec = manifest['specification']
    op.require(spec['phase'] == SCHEMA and spec['plan'] == plan
               and spec['plan_sha256'] == data.sha(plan_path)
               and manifest['code_sha256'] == plan['code_sha256'], 'Readiness run binding differs')
    op.require(result['complete'] and result['passed'], 'Readiness did not pass both complete task blocks')
    files = [plan_path, manifest_path, output / 'results.json'] + sorted((output / 'requests').glob('*.json'))
    files += sorted((output / 'preflight_receipts').glob('*.json'))
    return {'schema': SCHEMA + '-receipt', 'passed': True, 'complete': True,
        'live': spec['live'], 'freeze_commit': spec['freeze_commit'],
        'candidate_fingerprint': plan['candidate_fingerprint'],
        'candidate_plan_sha256': plan['candidate_plan_sha256'], 'plan_path': str(plan_path),
        'code_sha256': plan['code_sha256'], 'accounting': result['accounting'],
        'files_sha256': plan['source_files_sha256'] | {str(p): data.sha(p) for p in files}}


def verify_receipt(prep_dir, candidate_plan):
    """Replay every raw request and pass gate before admitting a final main plan."""
    output = Path(prep_dir).resolve()
    stored = json.loads((output / 'readiness-receipt.json').read_text())
    plan_path = Path(stored['plan_path']).resolve()
    plan = verify_plan(plan_path)
    op.require(op.fingerprint(candidate_plan) == plan['candidate_fingerprint'], 'Readiness candidate differs')
    _safe_output(plan, plan_path, output)
    with runtime.run_lock(output):
        journal = runtime.Journal(output, None, plan['limits'], HERE)
        result = collect_results(plan, journal)
        op.require(json.loads((output / 'results.json').read_text()) == result, 'Saved readiness result differs from replay')
        receipt = _receipt(plan, plan_path, output, result)
        op.require(receipt == stored, 'Readiness receipt or bound evidence changed')
    return receipt | {'files_sha256': receipt['files_sha256'] |
                      {str(output / 'readiness-receipt.json'): data.sha(output / 'readiness-receipt.json')}}


def execute(plan_path, output, client=None, *, freeze_commit=None, approved_plan_sha256=None,
            approve_live=False, preflight_receipt_dir=None):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = verify_plan(plan_path)
    _safe_output(plan, plan_path, output)
    live, receipt = client is None, None
    if live:
        verify_live_freeze(plan_path, approved_plan_sha256, freeze_commit, approve_live)
        op.require(preflight_receipt_dir is not None, 'Fresh route and allowance receipt required')
        from second_generator_preflight import verify_receipt as verify_preflight
        receipt = verify_preflight(plan_path, preflight_receipt_dir, expected_freeze_commit=freeze_commit)
        client = FrozenClient(plan, plan_path, receipt)
    specification = {'phase': SCHEMA, 'plan_sha256': data.sha(plan_path), 'plan': plan,
        'snapshots': plan['snapshots'], 'live': live, 'freeze_commit': freeze_commit if live else None}
    with runtime.run_lock(output):
        runtime.bind_run(output, specification)
        journal = runtime.Journal(output, client, plan['limits'], HERE)
        previous = collect_results(plan, journal)
        if previous['complete']:
            if previous['passed'] and not (output / 'readiness-receipt.json').exists():
                op.write_json(output / 'readiness-receipt.json', _receipt(plan, plan_path, output, previous))
            return previous
        journal.check_budget(0)
        if live:
            (output / 'preflight_receipts').mkdir(exist_ok=True)
            op.write_json(output / 'preflight_receipts' / (op.fingerprint(receipt) + '.json'), receipt)
            client.prepare()
        try:
            for job in plan['jobs']:
                concrete = materialize_job(plan, job, {r['id']: r for r in journal.records()})
                op.require(concrete is not None, 'Readiness stopped: generation did not pass technical gates')
                body, metadata = concrete
                attempt = journal.call(job['id'], plan['routes'][job['route']], body, metadata,
                    parse=parse_adherence if job['route'] == 'primary' else None, retries=1)
                if job['route'] == 'generator':
                    op.require(not _technical_errors(attempt), 'Readiness stopped: generation did not pass technical gates')
                else:
                    op.require(attempt['status'] == 'ok', 'Readiness stopped: invalid adherence judgment after allowed retry')
        finally:
            result = collect_results(plan, journal)
            op.write_json(output / 'results.json', result)
            if live:
                client.verify_bindings()
        if result['passed']:
            op.write_json(output / 'readiness-receipt.json', _receipt(plan, plan_path, output, result))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'receipt'))
    parser.add_argument('--candidate-plan')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output')
    parser.add_argument('--freeze-commit')
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--preflight-receipt-dir')
    parser.add_argument('--approve-live', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            op.require(args.candidate_plan and not Path(args.plan).exists(), 'Prepare needs a candidate and new plan path')
            value = build_readiness_plan(args.candidate_plan)
            op.write_json(args.plan, value)
        elif args.command == 'verify':
            value = verify_plan(args.plan)
        elif args.command == 'receipt':
            from second_generator_experiment import verify_candidate_plan
            plan = verify_plan(args.plan)
            value = verify_receipt(args.output, verify_candidate_plan(plan['candidate_plan_path']))
        else:
            value = execute(args.plan, args.output, freeze_commit=args.freeze_commit,
                approved_plan_sha256=args.approved_plan_sha256, approve_live=args.approve_live,
                preflight_receipt_dir=args.preflight_receipt_dir)
        print(json.dumps({key: value[key] for key in ('schema', 'initial_requests', 'maximum_attempts',
            'limits', 'passed', 'complete', 'accounting') if key in value} | {'plan_sha256': data.sha(args.plan)}))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError):
        print('Readiness stopped; preserve and inspect local evidence. No output replacement is permitted.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
