"""Original-plan OpenRouter study: offline planning, qualification, headroom and main runs.

Every live command needs --execute. Configuration templates are deliberately
unexecutable until the study owner supplies routes, snapshots, budgets and frozen gates.
The engineering smoke CLI remains separately versioned in api_runner.py.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

import api_runner as op
import api_data as data
import api_evaluation as ev
import runner_foundation_v1 as foundation
from research_client import Client
from api_env import api_key
from api_runtime import Journal, bind_run, run_lock, code_hashes

PROTOCOL = 'estr-openrouter-research-v1'
HERE = Path(__file__).resolve().parent


def route_check(route, judge=False):
    required = {'label', 'model', 'provider', 'family', 'reserve_usd', 'endpoint_snapshot'}
    if judge:
        required |= {'parameters', 'json_schema'}
    op.keys(route, required, 'route')
    op.require(op.identifier(route['label']) and op.identifier(route['family']), 'Invalid route/family label')
    op.require(isinstance(route['model'], str) and '/' in route['model'], 'Use a complete model ID')
    op.require(not any(v in route['model'].lower() for v in (':free', ':nitro', ':floor', 'latest', 'auto')), 'Changing routes are forbidden')
    op.require(isinstance(route['provider'], str) and route['provider'].strip(), 'Pin provider tag')
    op.require(op.positive(route['reserve_usd']), 'Reserve must be finite and positive')
    if judge:
        parameters_check(route['parameters'])
        op.require(type(route['json_schema']) is bool, 'Declare endpoint JSON-schema support from its snapshot')


def check_params(params, where):
    op.require(isinstance(params, dict) and set(params) <= op.PARAMETERS, f"Unsupported {where} fields")
    op.require(op.integer(params.get("max_tokens")), f"Set a positive {where}.max_tokens")
    for key in set(params) - {"reasoning", "max_tokens", "seed"}:
        op.require(type(params[key]) in (int, float) and math.isfinite(params[key]), f"Invalid {where}.{key}")
    for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1), ("min_p", 0, 1)):
        if key in params:
            op.require(low <= params[key] <= high, f"Invalid {where}.{key} range")
    if "reasoning" in params:
        op.require(isinstance(params["reasoning"], dict), f"{where}.reasoning must be an explicit object")
    for key in ('top_k','seed'):
        if key in params:
            op.require(type(params[key]) is int and params[key]>=0, f'{where}.{key} must be a nonnegative integer')


def parameters_check(params):
    check_params(params, "parameters")


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    op.keys(cfg, {'protocol', 'approved_by', 'decision_reference', 'freeze_commit', 'freeze_file', 'generators', 'judges', 'arms', 'generation',
                  'repetitions', 'seed', 'grouping_gates', 'quality_guard', 'judge_preference', 'limits'}
             | (set(cfg) & {'template_notes', 'analysis', 'evaluation_profile'}), 'research configuration')
    ev.profile(cfg.get('evaluation_profile'))
    if 'template_notes' in cfg:
        op.require(isinstance(cfg['template_notes'],list) and all(isinstance(note,str) for note in cfg['template_notes']), 'Template notes must be text')
    op.require(cfg['protocol'] == PROTOCOL, 'Wrong research protocol version')
    op.require(isinstance(cfg['approved_by'], list) and set(cfg['approved_by']) <= {'study_owner', 'second_contributor'}, 'Invalid approver list')
    op.require(isinstance(cfg['generators'], list) and cfg['generators'], 'At least one generator')
    op.require(isinstance(cfg['judges'], list) and cfg['judges'], 'At least one judge candidate')
    for judge, routes in ((False, cfg['generators']), (True, cfg['judges'])):
        for route in routes:
            route_check(route, judge)
        op.require(len({r['label'] for r in routes}) == len(routes), 'Duplicate route labels')
    op.require(not ({g['family'] for g in cfg['generators']} & {j['family'] for j in cfg['judges']}), 'Judge and generator families must differ')
    op.require(len({j['family'] for j in cfg['judges']}) == len(cfg['judges']), 'One candidate per judge family')
    op.require(set(cfg['judge_preference']) == {j['label'] for j in cfg['judges']} and len(cfg['judge_preference']) == len(cfg['judges']), 'Order every judge once')
    parameters_check(cfg['generation'])
    op.require(type(cfg['seed']) is int and cfg['seed'] >= 0, 'Nonnegative seed required')
    op.require(cfg['repetitions'] == 2 and type(cfg['repetitions']) is int, 'Original main plan has two repetitions')
    op.require(isinstance(cfg['arms'], list), 'Arms must be a list')
    ids = set()
    for arm in cfg['arms']:
        op.keys(arm, {'id', 'instruction', 'decoding_overrides'}, 'arm')
        op.require(op.identifier(arm['id']) and arm['id'] not in ids and isinstance(arm['instruction'], str), 'Invalid or duplicate arm')
        ids.add(arm['id'])
        op.require(isinstance(arm['decoding_overrides'], dict), 'Arm overrides must be an object')
        params = cfg['generation'] | arm['decoding_overrides']
        parameters_check(params)
        op.require(params['max_tokens'] == cfg['generation']['max_tokens'], 'Match generation token caps across arms')
    expected = {'default', 'naive_creative', 'manual_causal_shift_v1', 'manual_alternative_selection_v1'}
    op.require(expected <= ids, 'Keep default, naive and both manual arms')
    op.require(next(a for a in cfg['arms'] if a['id'] == 'default') == {'id': 'default', 'instruction': '', 'decoding_overrides': {}}, 'Default must be unmodified')
    for arm in cfg['arms']:
        if arm['instruction']:
            op.require(not arm['decoding_overrides'], 'Do not bundle instruction and decoding interventions')
    op.keys(cfg['limits'], {'max_requests', 'max_cost_usd', 'timeout_seconds'}, 'limits')
    op.require(op.integer(cfg['limits']['max_requests']) and op.positive(cfg['limits']['max_cost_usd']) and op.positive(cfg['limits']['timeout_seconds']), 'Invalid budgets')
    op.require(cfg['limits']['max_cost_usd'] < 50, 'Per-run max_cost_usd must stay below the illustrative US$50 cap')
    op.keys(cfg['grouping_gates'], {'ten', 'twenty'}, 'phase gates')
    from api_report import check_analysis
    check_analysis(cfg)
    return cfg, path.parent


def signature(cfg):
    evaluation = ev.profile(cfg.get('evaluation_profile'))
    return op.fingerprint({'protocol': PROTOCOL, 'judges': cfg['judges'],
                           'generator_families': [r['family'] for r in cfg['generators']],
                           'evaluator': evaluation['version'], 'prompts': evaluation['prompts'], 'preference': cfg['judge_preference']})


def generation_signature(cfg):
    return op.fingerprint({'generators': cfg['generators'], 'generation': cfg['generation'], 'seed': cfg['seed']})


def ready(cfg):
    op.require('study_owner' in cfg['approved_by'], 'the study owner approves configuration before paid execution')
    op.require(isinstance(cfg['decision_reference'], str) and cfg['decision_reference'].strip(), 'Record the frozen protocol decision')
    for route in cfg['generators'] + cfg['judges']:
        op.require('replace' not in (route['model'] + route['provider']).lower(), 'Replace template routes')


def route_snapshots(cfg, base, execute):
    result = {}
    for route in cfg['generators']:
        for arm in cfg['arms']:
            params = cfg['generation'] | arm['decoding_overrides']
            result[f"g-{route['label']}-{arm['id']}"] = data.endpoint_check(route, params, base, execute)
    for route in cfg['judges']:
        result['j-' + route['label']] = data.endpoint_check(route, route['parameters'], base, execute)
    return result


def planner(cfg, n=10):
    op.require(n in (10, 20), 'Choose n=10 or n=20 after headroom')
    models, arms, repetitions, tasks, checks = len(cfg['generators']), len(cfg['arms']), cfg['repetitions'], 10, 3
    cells = models * tasks * arms * repetitions
    generation = cells * n
    # One grouping + one quality call per response. A separate quality-only
    # primary costs the same number of calls. Check blocks cover all arms/reps.
    primary = cells * (n + 1)
    check = models * checks * arms * repetitions * (n + 1)
    generators = max(r['reserve_usd'] for r in cfg['generators'])
    judges = max(r['reserve_usd'] for r in cfg['judges'])
    return {'n': n, 'cells': cells, 'generation_calls': generation, 'primary_judge_calls': primary,
            'check_judge_calls': check, 'main_initial_calls': generation + primary + check,
            'main_max_calls_one_judge_retry': 2*generation + 2*(primary + check),
            'generation_no_reply_attempt_limit': 2,
            'main_reserved_usd_with_retries': 2*generation*generators + 2*(primary + check)*judges,
            'qualification_10_calls_per_candidate': 12*4 + 24*4 + 2*4,
            'manual_screen_proposal_calls': 2*2*5*2,
            'qualification_20_calls_per_grouping_judge': 4*4,
            'headroom_10_initial_calls': models*10*(10+1),
            'headroom_20_additional_initial_calls': models*10*(20+1),
            'headroom_10_max_calls': models*10*(2*10+2),
            'headroom_20_max_calls': models*10*(2*20+2),
            'fixture_construction': 'Authoring fixture budget; count replacement calls separately',
            'warning': 'Reserves are estimates. Set a hard key/account limit; endpoint smoke and search are additional.'}


def checked_fixtures(path, report_path, n):
    tool = data.load_editor_tool()
    path = Path(path)
    fixtures = json.loads(path.read_text())
    sets, pairs = tool.validate(fixtures)
    report = json.loads(Path(report_path).read_text())
    op.require(report['fixture_manifest']['sha256'] == data.sha(path), 'Checks refer to different fixture text')
    op.require(all(not x for x in report['unfinished'].values()), 'Fixture checks unfinished')
    op.require(report['outlines'] and all(x == 'pass' for x in report['outlines'].values()), 'Outlines not accepted')
    op.require(report['checks'] and all(x == 'pass' for x in report['checks'].values()), 'Fixture responses/pairs not accepted')
    # Exact expected rosters prevent substituting an unrelated report with only
    # one accepted item and the right manually supplied hash.
    expected_rows = [r for rater in tool.RATERS for r in tool.rows_for(sets, pairs, rater, 'not-used')]
    op.require(set(report['outlines']) == {r['item'] for r in expected_rows if r['kind'] == 'outlines'}, 'Outline check roster differs')
    op.require(set(report['checks']) == {r['item'] for r in expected_rows if r['kind'] != 'outlines'}, 'Response/pair check roster differs')
    op.require(set(report['unfinished']) == set(tool.RATERS), 'Missing checker completion record')
    scored = [s for s in sets if 'exploratory' not in s.get('description', '').lower()]
    exploratory = [s for s in sets if s not in scored]
    if n == 10:
        op.require(len(scored) == 12 and len(exploratory) == 2 and len(pairs) == 24, 'Qualification requires 12 scored + 2 exploratory sets and 24 pairs')
        op.require({s['task_id'] for s in scored} >= {t['id'] for t in data.development_tasks()}, 'Include all ten development tasks')
        mix = ((1, 1, 1), (3, 2, 3), (3, 4, 6), (4, 7, 9), (1, 10, 10))
        categories = {c: sum(p['category'] == c for p in pairs) for c in tool.CATEGORIES}
        op.require(all(v == 4 for v in categories.values()), 'Four pairs per category required')
    else:
        op.require(len(scored) == 4 and not pairs, '20-response qualification needs four grouping sets, no quality pairs')
        mix = ((1, 3, 4), (1, 8, 10), (1, 14, 17), (1, 20, 20))
    op.require(len({s['task_id'] for s in scored}) == len(scored), 'Scored tasks must differ')
    for s in sets:
        op.require(len(s['responses']) == n, f'Every grouping set requires {n} checked responses')
    ks = [len({r['family'] for r in s['responses']}) for s in scored]
    for count, lo, hi in mix:
        op.require(sum(lo <= k <= hi for k in ks) == count, 'Fixture count mix differs from frozen protocol')
    return fixtures, scored, exploratory, pairs


def qualification(cfg, base, fixture_path, checks_path, output, client, n=10, selected=None, adjudication_path=None):
    gates = cfg['grouping_gates']['ten' if n == 10 else 'twenty']
    ev.check_gates(gates, n)
    input_hashes = {str(path): data.sha(path) for path in (fixture_path, checks_path)}
    if adjudication_path is not None:
        input_hashes[str(adjudication_path)] = data.sha(adjudication_path)
    fixtures, sets, exploratory, pairs = checked_fixtures(fixture_path, checks_path, n)
    profile_id = cfg.get('evaluation_profile')
    evaluation = ev.profile(profile_id)
    amendment = None
    scoring_sets = sets
    if adjudication_path is not None:
        import adjudication
        op.require(n == 10, 'Adjudicated development qualification supports n=10 only')
        op.require(profile_id == ev.TASK_SENSITIVE_PROFILE, 'Adjudication requires the versioned task-sensitive profile')
        amendment = adjudication.load(adjudication_path, fixture_path, checks_path, sets)
        scoring_sets = adjudication.scoring_sets(sets, amendment)
    else:
        op.require(profile_id in (None, ev.VERSION), 'Amended qualification requires an explicit adjudication artifact')
    candidates = cfg['judges']
    if n == 20:
        op.require(selected and selected['status'] == 'selected', 'Qualify ten-response judges first')
        labels = {selected['grouping'], selected['check']} - {None}
        candidates = [j for j in candidates if j['label'] in labels]
    specification = {'phase': f'qualification_{n}', 'config': cfg, 'fixtures': fixtures,
                     'checks_sha256': input_hashes[str(checks_path)], 'snapshots': route_snapshots(cfg, base, True)}
    if amendment is not None:
        op.require(all(data.sha(path) == digest for path, digest in input_hashes.items()), 'Qualification inputs changed during validation')
        specification.update(adjudication=amendment, adjudication_sha256=input_hashes[str(adjudication_path)],
                             evaluation=evaluation, development_only=True)
    with run_lock(output):
        bind_run(output, specification)
        journal = Journal(output, client, cfg['limits'], base)
        summaries, sensitivities = {}, {}
        for candidate in candidates:
            groups, quality, extra = {}, {}, {}
            for entry in sets + exploratory:
                sid = entry['set_id']
                texts = [r['text'] for r in entry['responses']]
                orders = [ev.shuffle(n, cfg['seed'], [sid, i]) for i in range(3)]
                orders.append(orders[0])
                labels = []
                for repeat, order in enumerate(orders):
                    body = ev.judge_body(candidate, entry['task'], texts, 'grouping', order, profile_id=profile_id)
                    attempt = journal.call(f"group-{candidate['label']}-{sid}-{repeat}", candidate, body,
                                           {'role': 'grouping', 'set_id': sid, 'repeat': repeat, 'order': order,
                                            'evaluator': evaluation['version']}, parse=lambda text, order=order: ev.parse_grouping(text, order))
                    labels.append(attempt.get('parsed'))
                (groups if entry in sets else extra)[sid] = labels
            for pair in pairs:
                outcomes = []
                for side in ('a', 'b'):
                    scores = []
                    for repeat in range(2):
                        attempt = journal.call(f"score-{candidate['label']}-{pair['pair_id']}-{side}-{repeat}", candidate,
                            ev.judge_body(candidate, pair['task'], [pair[side]], 'quality', profile_id=profile_id),
                            {'role': 'quality', 'pair_id': pair['pair_id'], 'side': side, 'repeat': repeat, 'evaluator': evaluation['version']}, parse=ev.parse_quality)
                        scores.append(attempt.get('parsed'))
                    outcomes.append(scores)
                quality[pair['pair_id']] = outcomes
            summary = ev.grouping_summary(scoring_sets, groups, gates['failed_count_penalty'])
            summaries[candidate['label']] = {'grouping': summary, 'grouping_pass': ev.grouping_pass(summary, gates),
                                             'quality': ev.quality_summary(pairs, quality) if pairs else None,
                                             'exploratory': extra}
            if amendment is not None:
                # Evaluate the same replies against the original outline labels.
                # This is a prespecified sensitivity, never a second selection try.
                original = ev.grouping_summary(sets, groups, gates['failed_count_penalty'])
                sensitivities[candidate['label']] = {'grouping': original,
                    'grouping_pass': ev.grouping_pass(original, gates)}
        if n == 10:
            chosen = ev.select_judges(summaries, candidates, cfg['generators'], gates, cfg['judge_preference'])
        else:
            chosen = {'status': 'qualified' if summaries[selected['grouping']]['grouping_pass'] else 'stop',
                      'grouping': selected['grouping'], 'check': selected['check'],
                      'check_qualified': bool(selected['check'] and summaries[selected['check']]['grouping_pass'])}
        result = {'phase': f'qualification_{n}', 'evaluation_signature': signature(cfg), 'gates': gates,
                  'fixtures_sha256': input_hashes[str(fixture_path)], 'summaries': summaries,
                  'selection': chosen, 'accounting': journal.account()}
        if amendment is not None:
            op.require(all(data.sha(path) == digest for path, digest in input_hashes.items()), 'Qualification inputs changed during execution')
            result.update(development_only=True, adjudication_sha256=input_hashes[str(adjudication_path)],
                          adjudication_reviewers=amendment['reviewers'], numerical_selection=chosen,
                          sensitivity={'reference': 'original_outline_families', 'summaries': sensitivities})
            result['selection'] = chosen | {'status': 'development_only' if chosen['status'] == 'selected' else 'stop'}
        op.write_json(Path(output) / 'qualification.json', result)
        return result


def read_selection(path, cfg):
    result = json.loads(Path(path).read_text())
    op.require(not result.get('development_only', False), 'Development qualification is not eligible for downstream stages; human adjudication remains separate')
    op.require(result['phase'] == 'qualification_10' and result['evaluation_signature'] == signature(cfg), 'Qualification evaluator differs from this config')
    op.require(result['gates'] == cfg['grouping_gates']['ten'], 'Qualification gates changed')
    op.require(result['selection']['status'] == 'selected', 'No qualified primary judges')
    return result['selection']


def check_twenty(path, cfg, selected):
    result = json.loads(Path(path).read_text())
    op.require(result['phase'] == 'qualification_20' and result['evaluation_signature'] == signature(cfg), 'Wrong twenty-response qualification')
    op.require(result['gates'] == cfg['grouping_gates']['twenty'] and result['selection']['status'] == 'qualified', 'Primary judge not qualified at twenty responses')
    op.require(result['selection']['grouping'] == selected['grouping'] and result['selection']['check'] == selected['check'], 'Twenty-response qualification has different selected judges')
    return result['selection']


def run_comparison(cfg, base, selection, output, client, n=10, phase='main', qualification20=None, *, stage_authorization=None, exploratory_authorization=None):
    if stage_authorization is not None or selection.get('status') == 'fresh_check_only':
        op.require(stage_authorization is not None and phase == 'headroom_10' and n == 10,
                   'Fresh-check adoption authorizes only headroom_10')
        op.require(len(cfg['generators']) == 1 and selection['grouping'] == selection['quality'] == 'gemini'
                   and selection['check'] is None, 'Fresh-check adoption permits one generator and Hydrogen only')
    if exploratory_authorization is not None or selection.get('status') == 'exploratory_only':
        from exploratory_main import assert_scope, selection as adopted_selection
        op.require(stage_authorization is None and exploratory_authorization is not None
                   and phase == 'main' and n == 10 and qualification20 is None,
                   'Exploratory adoption authorizes only its n=10 main comparison')
        assert_scope(cfg, exploratory_authorization, 'main')
        op.require(selection == adopted_selection(cfg), 'Exploratory judge selection differs')
        # Require the committed freeze here, not only in the command-line wrapper.
        from exploratory_main import freeze_payload as exploratory_freeze
        validate_freeze(cfg, base, exploratory_freeze(cfg, base, exploratory_authorization))
    profile_id = cfg.get('evaluation_profile')
    evaluation = ev.profile(profile_id)
    tasks = data.development_tasks()
    op.require(n in (10, 20), 'Response count must come from headroom')
    op.require(n == 10 or qualification20 is not None, 'Twenty-response grouping qualification required')
    if n == 20:
        op.require(qualification20['grouping'] == selection['grouping'], 'Twenty-response primary judge differs')
        selection = selection | {'check_qualified': selection['check_qualified'] and qualification20['check_qualified']}
    arms = cfg['arms'] if phase == 'main' else [a for a in cfg['arms'] if a['id'] == 'default']
    repetitions = cfg['repetitions'] if phase == 'main' else 1
    routes = {r['label']: r for r in cfg['judges']}
    # Freeze the task-block sample before any primary judgment is read.
    check_tasks = {g['label']: sorted(tasks, key=lambda t: op.fingerprint([cfg['seed'], g['label'], t['id']]))[:3] for g in cfg['generators']}
    # Adapt the approved protocol to the second contributor's scheduling implementation. Counts and
    # seed derivation stay unchanged; outcomes never influence execution order.
    schedule = foundation.schedule({'analysis': {'seed': cfg['seed']},
        'models': cfg['generators'], 'arms': arms, 'repetitions': repetitions,
        'samples_per_task': n}, {'tasks': tasks})
    specification = {'phase': phase, 'n': n, 'config': cfg, 'selection': selection, 'qualification20': qualification20, 'schedule': schedule,
                     'tasks': tasks, 'snapshots': route_snapshots(cfg, base, True),
                     'check_tasks': {g: [t['id'] for t in ts] for g, ts in check_tasks.items()}}
    if stage_authorization is not None:
        specification['stage_authorization'] = stage_authorization
    if exploratory_authorization is not None:
        specification['exploratory_authorization'] = exploratory_authorization
    with run_lock(output):
        manifest = bind_run(output, specification)
        if (stage_authorization is not None or exploratory_authorization is not None) and hasattr(client, 'prepare'):
            client.prepare()
        journal = Journal(output, client, cfg['limits'], base)
        rows = []
        generators = {g['label']: g for g in cfg['generators']}
        task_map = {t['id']: t for t in tasks}
        arm_map = {a['id']: a for a in arms}
        for block in schedule['blocks']:
            generator, task = generators[block['generator']], task_map[block['task']]
            generated_by_cell = {}
            for job in block['generation_order']:
                arm, repetition, index = arm_map[job['arm']], job['repetition'], job['sample']
                cell = {'generator': generator['label'], 'task': task['id'], 'arm': arm['id'], 'repetition': repetition}
                identity = op.fingerprint(cell)[:20]
                parameters = cfg['generation'] | arm['decoding_overrides']
                if 'seed' in parameters:
                    parameters = parameters | {'seed': int(op.fingerprint([cfg['seed'], phase, task['id'], repetition, index])[:8], 16) & 0x7fffffff}
                content = task['text'] + ('\n\nAdditional instruction:\n' + arm['instruction'] if arm['instruction'] else '')
                generated = generated_by_cell.setdefault((arm['id'], repetition), [None] * n)
                generated[index] = journal.call(f'g-{identity}-{index}', generator,
                    op.payload(generator, [{'role': 'user', 'content': content}], parameters),
                    {'role': 'generation', 'cell': cell, 'index': index, 'phase': phase})
            # Every arm is generated before any measurement in this task/model block.
            for job in block['measurement_order']:
                arm, repetition = arm_map[job['arm']], job['repetition']
                cell = {'generator': generator['label'], 'task': task['id'], 'arm': arm['id'], 'repetition': repetition}
                identity = op.fingerprint(cell)[:20]
                generated = generated_by_cell[(arm['id'], repetition)]
                row = {'cell': cell, 'samples': n, 'generation_complete': all(a['status'] == 'ok' for a in generated),
                       'generations': generated, 'judges': {}}
                jobs = [('primary', routes[selection['grouping']], routes[selection['quality']])]
                if phase == 'main' and selection['check'] and task in check_tasks[generator['label']]:
                    jobs.append(('check', routes[selection['check']], routes[selection['check']]))
                for role, grouping_route, quality_route in jobs:
                    labels, scores = None, [None] * n
                    if row['generation_complete']:
                        texts = [a['text'] for a in generated]
                        order = ev.shuffle(n, cfg['seed'], [phase, identity])
                        attempt = journal.call(f'group-{identity}-{role}', grouping_route,
                            ev.judge_body(grouping_route, task['text'], texts, 'grouping', order, profile_id=profile_id),
                            {'role': 'grouping', 'cell': cell, 'order': order, 'evaluator': evaluation['version']},
                            parse=lambda text, order=order: ev.parse_grouping(text, order), retries=1)
                        labels = attempt.get('parsed')
                        if phase == 'main':
                            for index, text in enumerate(texts):
                                attempt = journal.call(f'score-{identity}-{role}-{index}', quality_route,
                                    ev.judge_body(quality_route, task['text'], [text], 'quality', profile_id=profile_id),
                                    {'role': 'quality', 'cell': cell, 'index': index, 'evaluator': evaluation['version']}, parse=ev.parse_quality, retries=1)
                                scores[index] = attempt.get('parsed')
                    row['judges'][role] = {'labels': labels, 'scores': scores,
                        'grouping_complete': labels is not None, 'quality_complete': all(s is not None for s in scores),
                        'grouping_judge': grouping_route['label'], 'quality_judge': quality_route['label'],
                        'metrics': ev.group_metrics(labels, scores) if labels is not None else None}
                rows.append(row)
        result = {'phase': phase, 'n': n, 'evaluation_signature': signature(cfg),
                  'generation_signature': generation_signature(cfg), 'check_tasks': specification['check_tasks'],
                  'code_sha256': manifest['code_sha256'], 'rows': rows, 'accounting': journal.account()}
        if stage_authorization is not None:
            result.update(development_only=True, main_eligible=False,
                          stage_authorization_sha256=op.fingerprint(stage_authorization))
        if exploratory_authorization is not None:
            from exploratory_main import FLAGS
            result.update(**FLAGS, exploratory_authorization_sha256=op.fingerprint(exploratory_authorization))
        op.write_json(Path(output) / 'results.json', result)
        if phase != 'main':
            counts = {g['label']: [len(set(r['judges']['primary']['labels'])) if r['judges']['primary']['labels'] is not None else None
                                  for r in rows if r['cell']['generator'] == g['label']] for g in cfg['generators']}
            result['headroom'] = ev.headroom(counts, n)
            op.write_json(Path(output) / 'headroom.json', result)
        return result


def simulation(fixture_path, output, seed):
    """Expose consequences of synthetic errors; never choose gates from judge results."""
    fixture = json.loads(Path(fixture_path).read_text())
    sets, _ = data.load_editor_tool().validate(fixture)
    sets = [s for s in sets if 'exploratory' not in s.get('description', '').lower()]
    op.require(sets and all(s.get('responses') for s in sets), 'Constructed grouping responses required')
    n = len(sets[0]['responses'])
    op.require(n in (10,20) and all(len(s['responses'])==n for s in sets), 'Consistent fixture size required')
    scenarios, perturbations = {}, {}
    borderline = ('single_merge', 'single_split', 'count_cancelling')
    for name in ('gold', 'all_merged', 'all_split', 'unstable', 'one_failure_per_set') + borderline:
        outcomes = {}
        applied_sets = []
        for entry in sets:
            expected = ev.canonical([r['family'] for r in entry['responses']])
            if name == 'all_merged': runs = [[0]*n]*4
            elif name == 'all_split': runs = [list(range(n))]*4
            elif name == 'unstable': runs = [expected, [0]*n, list(range(n)), expected]
            elif name == 'one_failure_per_set': runs = [None, expected, expected, expected]
            elif name in borderline:
                changed, applied = ev.borderline_partition(expected, name)
                runs = [changed] * 4
                if applied:
                    applied_sets.append(entry['set_id'])
            else: runs = [expected]*4
            outcomes[entry['set_id']] = runs
        scenarios[name] = {str(penalty): ev.grouping_summary(sets,outcomes,penalty) for penalty in (n,n+1,2*n)}
        if name in borderline:
            perturbations[name] = {'applied_sets': applied_sets,
                                   'unchanged_sets': [s['set_id'] for s in sets if s['set_id'] not in applied_sets]}
    result = {'schema': 'fixture-gate-simulation-v2', 'fixture_sha256': data.sha(fixture_path),
              'seed': seed, 'n': n, 'scenarios': scenarios,
              'perturbations': perturbations,
              'note': 'Deterministic synthetic diagnostic only; seed is retained as provenance, not used to randomize these cases. Borderline patterns leave ineligible sets unchanged and list them explicitly. the study owner freezes tolerances, penalty, ranking and zero-denominator policy before paid qualification.'}
    op.write_json(Path(output),result)
    return result


def prior_headroom(path, cfg):
    op.require(path, 'The completed ten-response headroom result is required before stage twenty')
    result = json.loads(Path(path).read_text())
    op.require(result['phase'] == 'headroom_10' and result['evaluation_signature'] == signature(cfg)
               and result['generation_signature'] == generation_signature(cfg), 'Ten-response headroom setup differs')
    op.require(result['headroom']['next'] != 'incomplete', 'Ten-response headroom is incomplete; recover no-reply failures before considering stage twenty')
    op.require(all(r['complete'] for r in result['headroom']['models'].values())
               and result['headroom']['next'] == 'stage_20', 'Finish stage ten; use twenty only if the complete screen needs it')
    return result


def freeze_payload(cfg, base, selection_path, headroom_path, n, qualification20_path=None, screening_path=None):
    ready(cfg)
    from api_report import check_quality_guard, check_analysis
    check_analysis(cfg, require_frozen=True)
    check_quality_guard(cfg, require_support=True, tasks=[t['id'] for t in data.development_tasks()])
    chosen = read_selection(selection_path,cfg)
    from api_screen import check_screen
    screen_sha=check_screen(screening_path,cfg,chosen)
    head = json.loads(Path(headroom_path).read_text())
    op.require(head['evaluation_signature'] == signature(cfg) and head['generation_signature'] == generation_signature(cfg)
               and head['n'] == n and head['headroom']['pass'], 'A complete passing headroom result is required')
    if n == 20:
        check_twenty(qualification20_path,cfg,chosen)
    return {'schema': 'openrouter-protocol-freeze-v1', 'config': cfg | {'freeze_commit': None}, 'n': n,
            'selection_sha256': data.sha(selection_path), 'headroom_sha256': data.sha(headroom_path),
            'qualification20_sha256': data.sha(qualification20_path) if n == 20 else None,
            'manual_screen_sha256':screen_sha,
            'snapshots': route_snapshots(cfg,base,True), 'code_sha256': code_hashes()}


def validate_freeze(cfg, base, current, repository=None):
    """Compare with the COMMITTED freeze, not an editable ignored local config."""
    import subprocess
    root = Path(repository) if repository is not None else Path(subprocess.check_output(
        ['git','-C',str(HERE),'rev-parse','--show-toplevel'],text=True).strip())
    commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    op.require(not dirty and cfg['freeze_commit'] == commit, 'Commit reviewed code and frozen protocol; record freeze_commit and keep the tree clean')
    path = (Path(base) / cfg['freeze_file']).resolve()
    op.require(path.is_relative_to(root.resolve()), 'Freeze file must be inside the repository')
    try:
        frozen = json.loads(subprocess.check_output(['git','-C',str(root),'show',commit+':'+str(path.relative_to(root.resolve()))],
                            text=True,stderr=subprocess.DEVNULL))
    except subprocess.CalledProcessError:
        raise ValueError('Freeze file is not committed at freeze_commit') from None
    op.require(frozen == current, 'Configuration, decisions or input artifacts differ from the committed freeze')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'snapshot', 'simulate', 'screen', 'freeze', 'qualify', 'qualify20', 'headroom', 'run', 'report', 'reconcile'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--fixtures'); parser.add_argument('--checks'); parser.add_argument('--selection')
    parser.add_argument('--adjudication', help='Versioned semantic partitions for development qualification only')
    parser.add_argument('--qualification20'); parser.add_argument('--headroom')
    parser.add_argument('--screen-spec'); parser.add_argument('--screening')
    parser.add_argument('--n', type=int, choices=(10, 20), default=10)
    parser.add_argument('--route', help='Route label for public endpoint metadata snapshot')
    parser.add_argument('--request-id'); parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--cost', type=float); parser.add_argument('--note')
    parser.add_argument('--resolve-setup', action='store_true', help='Record correction of credentials/service setup without changing the frozen request')
    parser.add_argument('--embeddings', action='store_true', help='Compute canonical local MiniLM metrics, offline')
    args = parser.parse_args(argv)
    try:
        cfg, base = load_config(args.config)
        op.require(not args.adjudication or args.command == 'qualify', '--adjudication is only valid for n=10 qualification')
        if args.command == 'plan':
            print(json.dumps(planner(cfg, args.n), indent=2)); return 0
        op.require(args.output, '--output is required')
        if args.command == 'snapshot':
            route = next((r for r in cfg['generators'] + cfg['judges'] if r['label'] == args.route), None)
            op.require(route is not None, 'Choose --route from config labels')
            data.fetch_snapshot(route, args.output)
        elif args.command == 'simulate':
            op.require(args.fixtures, 'Simulation needs --fixtures')
            simulation(args.fixtures, args.output, cfg['seed'])
        elif args.command == 'freeze':
            op.require(args.selection and args.headroom, 'Freeze needs --selection and --headroom')
            destination = (base / cfg['freeze_file']).resolve()
            op.require(Path(args.output).resolve() == destination, '--output must match config.freeze_file')
            op.require(not destination.exists(), 'Freeze file exists; preserve it or assign a new version')
            op.write_json(destination,freeze_payload(cfg,base,args.selection,args.headroom,args.n,args.qualification20,args.screening))
        elif args.command == 'report':
            from api_report import write_report
            write_report(args.output, embeddings=args.embeddings)
        elif args.command == 'reconcile':
            op.require(args.request_id and args.cost is not None and args.note, 'Reconciliation needs request, cost and evidence note')
            with run_lock(args.output):
                Journal(args.output, None, cfg['limits']).reconcile(args.request_id, args.attempt, args.cost, args.note,args.resolve_setup)
        else:
            op.require(args.execute, 'Paid commands require --execute; plan and report are offline')
            ready(cfg)
            client = Client(api_key(), cfg['limits']['timeout_seconds'])
            if args.command in ('qualify', 'qualify20'):
                op.require(args.fixtures and args.checks, 'Qualification requires frozen --fixtures and --checks')
                selected = read_selection(args.selection, cfg) if args.command == 'qualify20' else None
                if args.command == 'qualify20':
                    prior_headroom(args.headroom, cfg)
                qualification(cfg, base, args.fixtures, args.checks, args.output, client,
                              20 if args.command == 'qualify20' else 10, selected, args.adjudication)
            else:
                op.require(args.selection, 'Frozen --selection qualification result required')
                selected = read_selection(args.selection, cfg)
                if args.command == 'screen':
                    from api_screen import screen
                    op.require(args.screen_spec,'Supply the approved --screen-spec')
                    screen(cfg,base,selected,args.screen_spec,args.output,client)
                    print(f'Saved manual screen: {args.output}')
                    return 0
                q20 = check_twenty(args.qualification20, cfg, selected) if args.n == 20 else None
                if args.command == 'headroom' and args.n == 20:
                    prior_headroom(args.headroom, cfg)
                if args.command == 'run':
                    op.require(args.headroom, 'A completed --headroom result is required')
                    validate_freeze(cfg,base,freeze_payload(cfg,base,args.selection,args.headroom,args.n,args.qualification20,args.screening))
                    head = json.loads(Path(args.headroom).read_text())
                    op.require(head['evaluation_signature'] == signature(cfg) and head['generation_signature'] == generation_signature(cfg), 'Headroom setup differs')
                    op.require(head['headroom']['pass'] and head['n'] == args.n, 'Headroom incomplete or failed')
                run_comparison(cfg, base, selected, args.output, client, args.n,
                               'main' if args.command == 'run' else f'headroom_{args.n}', q20)
        print(f'Saved {args.command}: {args.output}')
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        print(f'Error: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
