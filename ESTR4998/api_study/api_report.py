"""Offline, within-generator comparisons with complete paired task blocks.

Repetitions are averaged within a task, then tasks receive equal weight.
Intervals resample paired task differences, not individual responses. Missing
groupings and missing quality remain separate; losing >2/10 tasks is flagged.
No quality non-inferiority claim is made without a separately frozen guard.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import math
import statistics

import api_runner as op
import api_data as data
import api_evaluation as ev
from api_runtime import environment, code_hashes

CORE_FIELDS = ('distinct_count', 'effective_count', 'quality', 'quality_filtered_4',
               'quality_filtered_5', 'quality_filtered_6')
EMBEDDING_FIELDS = ('vendi_score', 'rke_score', 'semantic_diversity', 'distinct_1',
                    'distinct_2', 'exact_duplicate_rate')
EXECUTION_FLAGS = {'exploratory': True, 'confirmatory': False, 'main_eligible': False}


def execution_provenance(manifest, result):
    """Never infer confirmatory status from a legacy main-phase filename.

    A new exploratory execution must carry the same explicit status and bound
    authorization in its manifest and result. Missing or contradictory markers
    are errors, not permission to fall back to the historical reporting path.
    This validates saved bindings, without treating them as independent evidence
    that the original scientific gates passed.
    """
    spec = manifest['specification']
    authorization = spec.get('exploratory_authorization')
    marked = any(key in result for key in (*EXECUTION_FLAGS, 'exploratory_authorization_sha256'))
    if 'exploratory_authorization' not in spec and not marked:
        return {'status': 'legacy_unspecified', **dict.fromkeys(EXECUTION_FLAGS),
                'authorization_sha256': None, 'authorization': None}
    op.require(isinstance(authorization, dict), 'Exploratory result requires its manifest authorization')
    for key, expected in EXECUTION_FLAGS.items():
        op.require(authorization.get(key) is expected and result.get(key) is expected,
                   f'Exploratory manifest/result {key} missing or inconsistent')
    op.require(result.get('development_only') is True, 'Exploratory result must retain development_only')
    adoption = authorization.get('adoption')
    op.require(isinstance(adoption, dict) and adoption.get('schema') == 'estr-exploratory-adoption-v1'
               and adoption.get('stage') == 'main', 'Exploratory main adoption required')
    op.require(spec.get('phase') == result.get('phase') == 'main', 'Exploratory main phases differ')
    cfg = spec['config']
    # Only the commit back-reference is normalized, as in the execution freeze.
    # In particular, score margins/analysis must not be editable at report time.
    op.require(adoption.get('config_sha256') == op.fingerprint(cfg | {'freeze_commit': None}),
               'Exploratory report configuration differs from the adopted configuration')
    digest = op.fingerprint(authorization)
    op.require(result.get('exploratory_authorization_sha256') == digest,
               'Exploratory authorization hash differs from the manifest')
    from study_runner import signature, generation_signature
    actual_signatures = {'evaluation_signature': signature(cfg), 'generation_signature': generation_signature(cfg)}
    for key, expected in actual_signatures.items():
        op.require(authorization.get(key) == result.get(key) == expected, f'Exploratory {key} differs')
    op.require(isinstance(manifest.get('code_sha256'), dict) and bool(manifest['code_sha256'])
               and result.get('code_sha256') == manifest['code_sha256'] == adoption.get('code_sha256'),
               'Exploratory execution code binding differs')
    op.require(adoption.get('development_tasks_sha256') == op.fingerprint(spec['tasks']),
               'Exploratory report task roster differs from the adopted roster')
    scope = adoption.get('scope')
    op.require(isinstance(scope, dict) and result.get('n') == spec.get('n') == scope.get('n'),
               'Exploratory report sample count differs from the adopted scope')
    op.require(spec.get('check_tasks') == result.get('check_tasks') ==
               {scope.get('generator'): scope.get('check_tasks')}, 'Exploratory check-task blocks differ')
    selected = spec['selection']
    op.require(selected.get('status') == 'exploratory_only'
               and all(key in scope and key in selected and selected[key] == scope[key]
                       for key in ('grouping', 'quality', 'check', 'check_qualified')),
               'Exploratory judge-role metadata differs from the adopted scope')
    return {'status': 'explicit_exploratory', **EXECUTION_FLAGS,
            'authorization_sha256': digest, 'authorization': authorization}


def check_analysis_reporting(reporting, tasks):
    """Validate a frozen subset declaration, never select tasks from outcomes."""
    op.keys(reporting, {'primary_task_ids', 'secondary_subsets'}, 'analysis reporting')
    primary = reporting['primary_task_ids']
    op.require(isinstance(primary, list) and all(isinstance(t, str) for t in primary)
               and len(primary) == len(set(primary)) == len(tasks) and set(primary) == set(tasks),
               'Primary reporting must retain every frozen task exactly once')
    subsets = reporting['secondary_subsets']
    op.require(isinstance(subsets, list), 'Secondary subsets must be a list')
    seen = set()
    for subset in subsets:
        op.keys(subset, {'id', 'tasks', 'selection_basis', 'measures', 'pooled_interval'}, 'secondary subset')
        op.require(isinstance(subset['id'], str) and subset['id'].strip() and subset['id'] not in seen,
                   'Unique secondary subset identifier required')
        seen.add(subset['id'])
        selected = subset['tasks']
        op.require(isinstance(selected, list) and selected and all(isinstance(t, str) for t in selected)
                   and len(selected) == len(set(selected)) and set(selected) <= set(tasks),
                   'Secondary subset must name unique frozen tasks')
        op.require(isinstance(subset['selection_basis'], str) and subset['selection_basis'].strip(),
                   'Record the secondary subset selection basis')
        measures = subset['measures']
        op.require(isinstance(measures, list) and measures and all(isinstance(m, str) for m in measures)
                   and len(measures) == len(set(measures)) and set(measures) <= set(CORE_FIELDS + EMBEDDING_FIELDS),
                   'Secondary subset measures must be declared reporting measures')
        op.require(subset['pooled_interval'] is False, 'Secondary subsets support per-task results only')
    return reporting


def secondary_task_effects(comparisons, reporting, cfg, analysis):
    """Expose selected task effects and missingness without a subset mean/CI.

    Values already average the declared repetitions within each task. The
    separate full-roster comparisons retain their original task weighting.
    """
    by_key = {(c['generator'], c['arm'], c['reference'], c['measure']): c
              for c in comparisons if c['role'] == 'primary'}
    rows = []
    for subset in reporting['secondary_subsets']:
        for model in cfg['generators']:
            for reference in analysis['references']:
                for arm in cfg['arms']:
                    if arm['id'] == reference:
                        continue
                    for measure in subset['measures']:
                        contrast = by_key.get((model['label'], arm['id'], reference, measure))
                        values = dict(zip(contrast['tasks_used'], contrast['per_task_differences'])) if contrast else {}
                        for task in subset['tasks']:
                            rows.append({'subset_id': subset['id'], 'selection_basis': subset['selection_basis'],
                                'generator': model['label'], 'arm': arm['id'], 'reference': reference,
                                'measure': measure, 'task': task, 'difference': values.get(task),
                                'availability': ('complete_pair' if task in values else
                                    'missing_complete_pair' if contrast else 'measure_not_computed')})
    return rows


def check_analysis(cfg, require_frozen=False):
    """References and primary contrasts are scientific choices, fixed before run.

    Missing analysis metadata remains readable for historical default-only
    reports. A new freeze must explicitly choose its comparisons; the reporter
    never selects the most favourable reference after seeing results.
    """
    analysis = cfg.get('analysis')
    if analysis is None:
        op.require(not require_frozen, 'Freeze an explicit analysis specification')
        return {'references': ['default'], 'primary_contrasts': [], 'decision_reference': None,
                'status': 'legacy_default_only_not_prespecified_here'}
    op.keys(analysis, {'references', 'primary_contrasts', 'decision_reference'}, 'analysis')
    arms = {a['id'] for a in cfg['arms']}
    references = analysis['references']
    op.require(isinstance(references, list) and references and all(isinstance(x, str) for x in references)
               and len(set(references)) == len(references) and set(references) <= arms,
               'Analysis references must name distinct declared arms')
    op.require('default' in references, 'Keep default contrasts alongside direct comparators')
    op.require(isinstance(analysis['primary_contrasts'], list), 'Primary contrasts must be a list')
    seen = set()
    for contrast in analysis['primary_contrasts']:
        op.keys(contrast, {'arm', 'reference', 'measure'}, 'primary contrast')
        op.require(all(isinstance(v, str) for v in contrast.values()), 'Contrast values must be names')
        key = (contrast['arm'], contrast['reference'], contrast['measure'])
        op.require(contrast['arm'] in arms and contrast['reference'] in references
                   and contrast['arm'] != contrast['reference'], 'Invalid primary arm/reference contrast')
        op.require(contrast['measure'] in CORE_FIELDS + EMBEDDING_FIELDS, 'Unknown primary measure')
        op.require(key not in seen, 'Duplicate primary contrast')
        seen.add(key)
    op.require(analysis['decision_reference'] is None or isinstance(analysis['decision_reference'], str),
               'Analysis decision reference must be text')
    resolved = bool(seen and analysis['decision_reference'] and analysis['decision_reference'].strip())
    op.require(not require_frozen or resolved, 'Freeze primary contrasts and their analysis decision reference')
    return analysis | {'status': 'declared_in_run_configuration' if resolved else 'pending'}


def metric_value(row, role, field):
    if row is None:
        return None
    if field == 'constraint_satisfaction':
        return row.get(field)
    if field in EMBEDDING_FIELDS:
        return row.get('embedding_metrics', {}).get(field)
    judge = row['judges'].get(role)
    if not judge:
        return None
    if field == 'quality':
        scores = judge['scores']
        return statistics.mean(scores) if judge['quality_complete'] and scores else None
    if not judge['grouping_complete']:
        return None
    if field.startswith('quality_filtered_'):
        return judge['metrics'][field]['distinct_count']
    return judge['metrics'][field]


def interval(values, seed, draws=5000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [samples[int(.025 * draws)], samples[min(draws-1, int(.975 * draws))]]


def paired_comparisons(rows, arms, repetitions, tasks, role='primary', field='distinct_count', seed=0, reference='default', models=None):
    """A task contributes only when both arms have every repetition.

    Reports pass the frozen generator roster so entirely missing generators
    still produce explicit unavailable contrasts and insufficient-support guards.
    The observed-roster default preserves callers without a frozen design.
    """
    results = []
    models = sorted({r['cell']['generator'] for r in rows} if models is None else models)
    by_cell = {}
    for row in rows:
        c = row['cell']
        key = (c['generator'], c['task'], c['arm'], c['repetition'])
        op.require(key not in by_cell, 'Duplicate result cell')
        by_cell[key] = row
    for model in models:
        for arm in arms:
            if arm == reference:
                continue
            differences, retained, missing = [], [], []
            for task in tasks:
                values = {}
                for condition in (reference, arm):
                    block = [by_cell.get((model, task, condition, repetition)) for repetition in range(repetitions)]
                    current = [metric_value(row, role, field) for row in block]
                    values[condition] = statistics.mean(current) if all(v is not None for v in current) else None
                if any(v is None for v in values.values()):
                    missing.append(task)
                else:
                    retained.append(task)
                    differences.append(values[arm] - values[reference])
            results.append({'generator': model, 'arm': arm, 'reference': reference, 'role': role, 'measure': field,
                            'tasks_used': retained, 'tasks_lost': missing, 'weak': len(missing) > 2,
                            'per_task_differences': differences,
                            'mean_difference': statistics.mean(differences) if differences else None,
                            'paired_task_bootstrap_95': interval(differences, f'{seed}-{model}-{arm}-{field}')})
    return results


def support_policy(guard, metric):
    return (guard.get('support_policies') or {}).get(metric) or guard.get('support_policy')


def eligible_tasks(tasks, metric):
    if metric != 'constraint_satisfaction':
        return list(tasks)
    constraints, _ = data.local_modules()
    return [t for t in tasks if t in constraints.PROMPT_CONSTRAINTS]


def check_quality_guard(cfg, require_support=False, tasks=None):
    guard=cfg['quality_guard']
    op.keys(guard,{'reference','margins','decision_reference'} | (set(guard) & {'support_policy', 'support_policies'}),'quality guard')
    op.require(isinstance(guard['margins'],dict) and 'quality' in guard['margins']
               and set(guard['margins']) <= {'quality','constraint_satisfaction'},'Declare judged quality and any constraint-rate guard separately')
    policies = guard.get('support_policies')
    op.require(policies is None or isinstance(policies, dict), 'Per-measure support policies must be an object')
    if policies is not None:
        op.require(set(policies) <= set(guard['margins']), 'Support policy without a declared quality measure')
    for policy in [guard.get('support_policy')] + list((policies or {}).values()):
        if policy is None:
            continue
        op.keys(policy, {'minimum_tasks', 'maximum_missing_tasks'}, 'quality support policy')
        op.require(type(policy['minimum_tasks']) is int and policy['minimum_tasks'] >= 2, 'Support requires at least two independent tasks')
        op.require(type(policy['maximum_missing_tasks']) is int and policy['maximum_missing_tasks'] >= 0, 'Missing-task allowance must be a nonnegative integer')
    op.require(guard['reference'] in {a['id'] for a in cfg['arms']},'Quality reference must be a declared arm')
    op.require(isinstance(guard['decision_reference'],str) and guard['decision_reference'].strip(),'Freeze a quality-guard decision before the main run')
    for metric,margin in guard['margins'].items():
        policy = support_policy(guard, metric)
        op.require(not require_support or policy is not None, f'Freeze an explicit quality-guard support policy for {metric}')
        if tasks is not None and policy is not None:
            eligible = eligible_tasks(tasks, metric)
            op.require(policy['minimum_tasks'] <= len(eligible),
                       f'{metric} minimum_tasks exceeds its {len(eligible)} eligible tasks')
            op.require(policy['maximum_missing_tasks'] <= len(eligible),
                       f'{metric} maximum_missing_tasks exceeds its eligible population')
        op.require(type(margin) in (int,float) and math.isfinite(margin) and margin >= 0,
                   f'Freeze a nonnegative margin for {metric}')
        if metric=='constraint_satisfaction':
            op.require(margin <= 1,'Constraint-rate margin must be in 0–1 units')


def noninferiority(bounds, margin):
    """Respect the inclusive margin despite score-mean roundoff.

    The 1e-12 absolute tolerance only absorbs floating-point arithmetic at
    the boundary; it does not enlarge the frozen scientific margin.
    """
    if bounds is None:
        return 'unavailable'
    if bounds[0] >= -margin or math.isclose(bounds[0], -margin, rel_tol=0, abs_tol=1e-12):
        return 'non_inferior'
    if bounds[1] < -margin and not math.isclose(bounds[1], -margin, rel_tol=0, abs_tol=1e-12):
        return 'inferior'
    return 'inconclusive'


def guard_verdict(contrast, margin, policy):
    """Task bootstrap needs replication; missingness policy is a frozen decision.

    Two tasks are a mathematical minimum, not a claim of sufficient power.
    Without a declared missingness policy, surviving-task intervals are only
    descriptive when tasks are lost. Keep the point estimate available.
    """
    used, lost = len(contrast['tasks_used']), len(contrast['tasks_lost'])
    if used < 2:
        return 'insufficient_support'
    if policy is None and lost:
        return 'support_policy_pending'
    if policy and (used < policy['minimum_tasks'] or lost > policy['maximum_missing_tasks']):
        return 'insufficient_support'
    return noninferiority(contrast['paired_task_bootstrap_95'], margin)


def missingness_diagnostic(contrast):
    """Bound the full eligible-task mean without assuming failures are random.

    A 1–10 score difference lies in [-9,9]; a rate difference in [-1,1].
    Assigning those extremes to every lost task bounds the *point mean* over
    the declared tasks. This is neither a confidence interval nor a new guard
    verdict, and it does not account for evaluator or finite-sample error.
    """
    used, lost = len(contrast['tasks_used']), len(contrast['tasks_lost'])
    extent = {'quality': 9, 'constraint_satisfaction': 1}[contrast['measure']]
    total = used + lost
    observed_sum = sum(contrast['per_task_differences'])
    bounds = [(observed_sum - lost * extent) / total, (observed_sum + lost * extent) / total] if total else None
    return {'verdict_scope': 'retained_complete_tasks' if lost else 'all_eligible_tasks',
            'full_eligible_task_mean_bounds': bounds,
            'missingness_note': 'Bounded point-mean sensitivity, not a confidence interval or replacement guard. '
                               'A passing retained-task guard does not establish preservation on missing tasks.'}


def request_costs(output, rows, accounting):
    """Allocate every journal attempt, including failed/retried judge calls.

    Result rows contain only each generation's final attempt, so summing them
    would understate cost. Missing journals leave costs unavailable; records
    without a matching result cell remain explicitly unallocated.
    """
    def empty():
        return {'attempted_requests': 0, 'known_cost_usd': 0.0, 'unknown_cost_attempts': 0,
                'committed_usd': 0.0}
    by_arm = defaultdict(empty)
    by_role = defaultdict(empty)
    unallocated, total = empty(), empty()
    cells = {tuple(r['cell'][k] for k in ('generator', 'task', 'arm', 'repetition')) for r in rows}
    paths = sorted((Path(output) / 'requests').glob('*.json'))
    hashes = {}
    def add(bucket, attempt):
        cost = attempt.get('reported_cost_usd')
        op.require(cost is None or (type(cost) in (int, float) and math.isfinite(cost) and cost >= 0), 'Invalid journal cost')
        reserve = attempt.get('reserve_usd')
        op.require(cost is not None or (type(reserve) in (int, float) and math.isfinite(reserve) and reserve >= 0), 'Unknown cost needs a recorded reserve')
        bucket['attempted_requests'] += 1
        bucket['known_cost_usd'] += cost or 0
        bucket['unknown_cost_attempts'] += cost is None
        bucket['committed_usd'] += cost if cost is not None else reserve
    for path in paths:
        record = json.loads(path.read_text())
        hashes[path.name] = data.sha(path)
        metadata = record.get('metadata', {})
        cell = metadata.get('cell', {})
        key = tuple(cell.get(k) for k in ('generator', 'task', 'arm', 'repetition'))
        for attempt in record['attempts']:
            add(total, attempt)
            if key not in cells:
                add(unallocated, attempt)
            else:
                add(by_arm[(cell['generator'], cell['arm'])], attempt)
                add(by_role[(cell['generator'], cell['arm'], metadata.get('role', 'unspecified'))], attempt)
    reconciles = bool(paths) and all(k in accounting and math.isclose(total[k], accounting[k], abs_tol=1e-9)
                                   for k in total)
    return {'status': 'complete' if reconciles else ('incomplete_journal' if paths else 'journal_unavailable'),
            'totals_from_journal': total if paths else None, 'matches_results_accounting': reconciles,
            'unallocated': unallocated if paths else None, 'request_sha256': hashes,
            'by_arm': [{'generator': g, 'arm': a, **v} for (g, a), v in sorted(by_arm.items())],
            'by_role': [{'generator': g, 'arm': a, 'role': role, **v}
                        for (g, a, role), v in sorted(by_role.items())]}


def arm_summaries(rows, cfg, tasks, n, costs, embeddings):
    """Descriptive arm means use complete repetitions and equal task weights.

    Different measures/arms can have different support. Comparative claims use
    the paired contrast table, never differences between these marginal means.
    Length/token distributions describe retained final replies, not all paid
    attempts; the separate request ledger accounts for every paid attempt.
    Words use whitespace splitting. Completion tokens are provider-reported
    usage and may include reasoning; they are not a count of visible words.
    """
    by_cell = {}
    for row in rows:
        c = row['cell']
        key = tuple(c[k] for k in ('generator', 'task', 'arm', 'repetition'))
        op.require(key not in by_cell, 'Duplicate result cell')
        by_cell[key] = row
    fields = CORE_FIELDS + (EMBEDDING_FIELDS if embeddings else ()) + ('constraint_satisfaction',)
    cost_map = {(c['generator'], c['arm']): c for c in costs['by_arm']}
    summaries = []
    for generator in cfg['generators']:
        model = generator['label']
        for arm in cfg['arms']:
            aid = arm['id']
            blocks = [by_cell.get((model, t, aid, r)) for t in tasks for r in range(cfg['repetitions'])]
            present = [r for r in blocks if r is not None]
            metrics = {}
            for field in fields:
                values, lost = {}, []
                eligible = eligible_tasks(tasks, field)
                for task in eligible:
                    samples = [metric_value(by_cell.get((model, task, aid, r)), 'primary', field)
                               for r in range(cfg['repetitions'])]
                    if all(v is not None for v in samples):
                        values[task] = statistics.mean(samples)
                    else:
                        lost.append(task)
                metrics[field] = {'mean': statistics.mean(values.values()) if values else None,
                                  'tasks_used': list(values), 'tasks_lost': lost,
                                  'per_task_values': values}
            attempts = [a for r in present for a in r['generations']]
            successful = [a for a in attempts if a['status'] == 'ok']
            scores = [s for r in present for s in r['judges']['primary']['scores'] if s is not None]
            usage = [(a.get('response') or {}).get('usage') or {} for a in successful]
            tokens = [u['completion_tokens'] for u in usage if type(u.get('completion_tokens')) is int and u['completion_tokens'] >= 0]
            words = [len(a['text'].split()) for a in successful]
            filters = {}
            for cutoff in (4, 5, 6):
                complete = [r['judges']['primary']['metrics'][f'quality_filtered_{cutoff}'] for r in present
                            if r['judges']['primary']['grouping_complete'] and r['judges']['primary']['quality_complete']]
                filters[str(cutoff)] = {'complete_cells': len(complete),
                    'retained_responses': sum(x['retained'] for x in complete),
                    'attempted_responses': sum(x['attempted'] for x in complete),
                    'represented_groups_sum': sum(x['distinct_count'] for x in complete),
                    'zero_acceptance_cells': sum(x['zero_acceptance'] for x in complete)}
            summaries.append({'generator': model, 'arm': aid, 'metrics': metrics,
                'expected_cells': len(blocks), 'observed_cells': len(present),
                'missing_cells': [{'task': t, 'repetition': r} for t in tasks for r in range(cfg['repetitions'])
                                  if (model, t, aid, r) not in by_cell],
                'generation_complete_cells': sum(r['generation_complete'] for r in present),
                'grouping_complete_cells': sum(r['judges']['primary']['grouping_complete'] for r in present),
                'quality_complete_cells': sum(r['judges']['primary']['quality_complete'] for r in present),
                'grouped_responses': sum(len(r['judges']['primary']['labels']) for r in present
                                         if r['judges']['primary']['grouping_complete']),
                'expected_responses': len(blocks) * n, 'observed_final_attempts': len(attempts),
                'successful_final_replies': len(successful), 'failed_final_attempts': len(attempts) - len(successful),
                'quality_scores_available': len(scores), 'quality_score_histogram': {str(s): scores.count(s) for s in range(1, 11)},
                'quality_filtered': filters,
                'mean_words_successful_replies': statistics.mean(words) if words else None,
                'mean_completion_tokens_with_usage': statistics.mean(tokens) if tokens else None,
                'completion_token_usage_available': len(tokens),
                'completion_tokens_successful_replies_total': sum(tokens) if tokens else None,
                'cost': cost_map.get((model, aid)) if costs['status'] != 'journal_unavailable' else None})
    return summaries


def default_saturation(rows, cfg, tasks, n):
    """Count fresh default cells, before averaging repetitions within tasks.

    Missing primary groupings are unavailable, not observations below ceiling.
    This diagnostic neither changes the paired contrasts nor selects tasks.
    """
    op.require(type(n) is int and n > 0, 'Saturation sample count must be a positive integer')
    if 'default' not in {a['id'] for a in cfg['arms']}:
        return []
    by_cell = {tuple(r['cell'][k] for k in ('generator', 'task', 'arm', 'repetition')): r for r in rows}
    summaries = []
    for generator in cfg['generators']:
        model = generator['label']
        counts = {'at_ceiling_cells': 0, 'one_below_ceiling_cells': 0, 'at_least_two_below_ceiling_cells': 0}
        missing = []
        for task in tasks:
            for repetition in range(cfg['repetitions']):
                row = by_cell.get((model, task, 'default', repetition))
                primary = row['judges'].get('primary') if row else None
                if not primary or not primary['grouping_complete']:
                    missing.append({'task': task, 'repetition': repetition,
                                    'reason': 'missing_result_cell' if row is None else 'incomplete_primary_grouping'})
                    continue
                labels = primary['labels']
                op.require(isinstance(labels, list) and len(labels) == n,
                           'Saturation grouping coverage differs from the response count')
                count = len(set(labels))
                field = ('at_ceiling_cells' if count == n else 'one_below_ceiling_cells' if count == n - 1
                         else 'at_least_two_below_ceiling_cells')
                counts[field] += 1
        expected = len(tasks) * cfg['repetitions']
        summaries.append({'generator': model, 'arm': 'default', 'sample_count': n,
                          'expected_cells': expected, 'available_cells': sum(counts.values()),
                          'unavailable_cells': len(missing), 'missing_cells': missing, **counts})
    return summaries


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], sort_keys=True) if isinstance(row.get(k), (list, dict)) else row.get(k)
                             for k in fields})


def export_tables(output, report):
    """CSV rows are contrasts, task effects, arm/measure means, score bins,
    cutoff summaries, and guards respectively. List/dict cells contain JSON;
    missing scalar values are empty, never substituted with numeric zero.
    """
    # Every table carries the run status so detached CSVs cannot silently lose
    # the distinction between exploratory execution and legacy unspecified data.
    provenance = report['execution_provenance']
    status = {'execution_status': provenance['status'],
              **{k: json.dumps(provenance[k]) if provenance[k] is not None else None for k in EXECUTION_FLAGS},
              'exploratory_authorization_sha256': provenance['authorization_sha256']}
    def table(name, fields, rows):
        write_csv(output / name, list(status) + fields, (row | status for row in rows))
    comparison_fields = ['generator', 'arm', 'reference', 'role', 'measure', 'analysis_role', 'mean_difference',
                         'paired_task_bootstrap_95', 'tasks_used', 'tasks_lost', 'weak']
    table('comparisons.csv', comparison_fields, report['comparisons'])
    per_task = [dict((k, c[k]) for k in comparison_fields[:6]) | {'task': t, 'difference': d}
                for c in report['comparisons'] for t, d in zip(c['tasks_used'], c['per_task_differences'])]
    table('per_task_effects.csv', comparison_fields[:6] + ['task', 'difference'], per_task)
    table('secondary_task_effects.csv', ['subset_id', 'selection_basis', 'generator', 'arm', 'reference',
          'measure', 'task', 'difference', 'availability'], report['secondary_task_effects'])
    flat = []
    for arm in report['arm_summaries']:
        for measure, value in arm['metrics'].items():
            flat.append({k: v for k, v in arm.items() if k not in ('metrics', 'cost', 'quality_filtered', 'quality_score_histogram')}
                        | {'measure': measure, **value, 'known_cost_usd': (arm['cost'] or {}).get('known_cost_usd'),
                           'cost_allocation_status': report['cost_allocation']['status']})
    table('arm_summary.csv', ['generator', 'arm', 'measure', 'mean', 'tasks_used', 'tasks_lost',
        'expected_cells', 'observed_cells', 'generation_complete_cells', 'grouping_complete_cells', 'quality_complete_cells',
        'expected_responses', 'observed_final_attempts', 'successful_final_replies', 'failed_final_attempts',
        'quality_scores_available', 'mean_words_successful_replies', 'mean_completion_tokens_with_usage',
        'completion_token_usage_available', 'known_cost_usd', 'cost_allocation_status'], flat)
    table('quality_distributions.csv', ['generator', 'arm', 'score', 'count'],
              [{'generator': a['generator'], 'arm': a['arm'], 'score': score, 'count': count}
               for a in report['arm_summaries'] for score, count in a['quality_score_histogram'].items()])
    table('quality_filtered.csv', ['generator', 'arm', 'cutoff', 'complete_cells', 'retained_responses',
              'attempted_responses', 'represented_groups_sum', 'zero_acceptance_cells'],
              [{'generator': a['generator'], 'arm': a['arm'], 'cutoff': cutoff, **values}
               for a in report['arm_summaries'] for cutoff, values in a['quality_filtered'].items()])
    table('quality_guards.csv', ['generator', 'arm', 'reference', 'measure', 'margin', 'mean_difference',
              'paired_task_bootstrap_95', 'tasks_used', 'tasks_lost', 'support_policy', 'verdict', 'verdict_scope',
              'full_eligible_task_mean_bounds', 'missingness_note', 'guard_interpretation'], report['quality_guards'])
    diagnostics = {(r['generator'], r['arm']): r for r in report['arm_diagnostics']}
    diagnostic_rows = []
    for arm in report['arm_summaries']:
        flags = diagnostics.get((arm['generator'], arm['arm']), {})
        diagnostic_rows.append({k: arm[k] for k in ('generator', 'arm', 'expected_responses', 'observed_final_attempts',
             'successful_final_replies', 'failed_final_attempts', 'quality_scores_available')}
             | {'unobserved_final_slots': arm['expected_responses'] - arm['observed_final_attempts'],
                'grouped_responses': arm['grouped_responses']}
             | {k: flags.get(k, 0) for k in ('diagnosed_replies', 'empty', 'refusal', 'mixed_script', 'truncated',
                                            'flag_free', 'constraint_applicable', 'constraint_passed')})
    table('diagnostics.csv', ['generator', 'arm', 'expected_responses', 'observed_final_attempts',
        'successful_final_replies', 'failed_final_attempts', 'unobserved_final_slots', 'quality_scores_available',
        'grouped_responses', 'diagnosed_replies', 'empty', 'refusal', 'mixed_script', 'truncated', 'flag_free',
        'constraint_applicable', 'constraint_passed'], diagnostic_rows)


def write_report(output, embeddings=False, embedder=None):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    cfg = manifest['specification']['config']
    analysis = check_analysis(cfg)
    op.require(analysis['status'] != 'pending', 'Resolve the stored analysis decisions before a main run; do not choose them after results')
    result = json.loads((output / 'results.json').read_text())
    op.require(result['phase'] == 'main', 'Main-run results required for comparison report')
    provenance = execution_provenance(manifest, result)
    tasks = [t['id'] for t in manifest['specification']['tasks']]
    if provenance['exploratory']:
        reporting = check_analysis_reporting(provenance['authorization']['adoption'].get('analysis_reporting'), tasks)
    else:
        reporting = {'primary_task_ids': tasks, 'secondary_subsets': []}
    rows, diagnostics, aggregate = result['rows'], [], defaultdict(lambda: {'attempted': 0, 'empty': 0, 'refusal': 0, 'mixed_script': 0,
        'truncated': 0, 'flag_free': 0, 'diagnosed_replies': 0, 'constraint_applicable': 0,
        'constraint_passed': 0, 'parsed_quality': 0, 'top_score': 0})
    if embeddings and embedder is None:
        _, metrics = data.local_modules()
        embedder = metrics.Embedder(data.EMBEDDING['model_id'], data.EMBEDDING['revision'], device='cpu', window=256)
    for row in rows:
        c = row['cell']
        op.require(row['samples'] == result['n'] and len(row['generations']) == result['n'],
                   'Result response count differs from the declared sample count')
        op.require(all(len(j['scores']) == result['n'] and
                       (not j['grouping_complete'] or len(j['labels']) == result['n'])
                       for j in row['judges'].values()), 'Judge coverage differs from the response count')
        counts = aggregate[(c['generator'], c['arm'])]
        flags = []
        for attempt in row['generations']:
            counts['attempted'] += 1
            if attempt['status'] != 'ok':
                flags.append(None)
                continue
            diagnostic = data.text_diagnostics(c['task'], attempt['text'], attempt['finish_reason'])
            flags.append(diagnostic)
            counts['diagnosed_replies'] += 1
            for flag in ('empty', 'refusal', 'mixed_script'):
                counts[flag] += diagnostic['flags'][flag]
            counts['truncated'] += diagnostic['truncated']
            counts['flag_free'] += not any(diagnostic['flags'][flag] for flag in ('empty', 'refusal', 'mixed_script')) and not diagnostic['truncated']
            counts['constraint_applicable'] += diagnostic['constraint_satisfied'] is not None
            counts['constraint_passed'] += diagnostic['constraint_satisfied'] is True
        scores = row['judges']['primary']['scores']
        applicable=[f['constraint_satisfied'] for f in flags if f is not None and f['constraint_satisfied'] is not None]
        row['constraint_satisfaction']=statistics.mean(applicable) if applicable and row['generation_complete'] else None
        counts['parsed_quality'] += sum(s is not None for s in scores)
        counts['top_score'] += sum(s == 10 for s in scores)
        entry = {'cell': c, 'diagnostics': flags,
                 'generation_complete': row['generation_complete'],
                 'generation_failures': [{'index': i, 'status': a['status'],
                     'failure_diagnostic': a.get('failure_diagnostic')}
                     for i, a in enumerate(row['generations']) if a['status'] != 'ok'],
                 'judge_coverage': {role: {'grouping_complete': judge['grouping_complete'],
                     'quality_complete': judge['quality_complete'],
                     'missing_quality_indices': [i for i, s in enumerate(judge['scores']) if s is None]}
                     for role, judge in row['judges'].items()}}
        if embeddings and row['generation_complete']:
            entry['canonical_embedding_metrics'] = data.embedding_metrics([a['text'] for a in row['generations']], embedder)
            row['embedding_metrics'] = entry['canonical_embedding_metrics']['values']
        if row['judges'].get('check'):
            primary, check = row['judges']['primary'], row['judges']['check']
            entry['evaluator_sensitivity'] = {
                'check_qualified': manifest['specification']['selection']['check_qualified'],
                'ari': ev.ari(primary['labels'], check['labels']) if primary['grouping_complete'] and check['grouping_complete'] else None,
                'count_difference': len(set(check['labels'])) - len(set(primary['labels'])) if primary['grouping_complete'] and check['grouping_complete'] else None,
                'mean_quality_difference': statistics.mean(check['scores']) - statistics.mean(primary['scores']) if primary['quality_complete'] and check['quality_complete'] else None}
        diagnostics.append(entry)
    arms = [a['id'] for a in cfg['arms']]
    models = [g['label'] for g in cfg['generators']]
    expected_cells = {(g['label'], t, a, r) for g in cfg['generators'] for t in tasks for a in arms for r in range(cfg['repetitions'])}
    observed = set()
    for row in rows:
        key = tuple(row['cell'][k] for k in ('generator', 'task', 'arm', 'repetition'))
        op.require(key in expected_cells, 'Result cell outside the frozen design')
        op.require(key not in observed, 'Duplicate result cell')
        observed.add(key)
    saturation = default_saturation(rows, cfg, tasks, result['n'])
    comparisons = []
    for role in ('primary', 'check'):
        # Check comparisons have a preselected three-task denominator, rather
        # than describing the seven deliberately unselected tasks as lost.
        for model in sorted(models):
            selected = tasks if role == 'primary' else result['check_tasks'][model]
            model_rows = [r for r in rows if r['cell']['generator'] == model]
            fields = list(CORE_FIELDS)
            if embeddings and role == 'primary':
                fields += EMBEDDING_FIELDS
            for field in fields:
                for reference in analysis['references']:
                    comparisons.extend(paired_comparisons(model_rows, arms, cfg['repetitions'], selected, role, field, cfg['seed'], reference, models=[model]))
    for comparison in comparisons:
        declared = {k: comparison[k] for k in ('arm', 'reference', 'measure')} in analysis['primary_contrasts']
        comparison['analysis_role'] = ('declared_primary' if declared and comparison['role'] == 'primary' else
                                       'sensitivity' if comparison['role'] == 'check' else 'exploratory')
    secondary = secondary_task_effects(comparisons, reporting, cfg, analysis)
    rates = []
    for (generator, arm), values in sorted(aggregate.items()):
        rates.append({'generator': generator, 'arm': arm, **values,
                      'final_attempt_denominator': values['attempted'],
                      'flag_free_share_of_diagnosed': values['flag_free'] / values['diagnosed_replies'] if values['diagnosed_replies'] else None,
                      'constraint_satisfaction_rate': values['constraint_passed'] / values['constraint_applicable'] if values['constraint_applicable'] else None,
                      'top_score_share_of_parsed': values['top_score'] / values['parsed_quality'] if values['parsed_quality'] else None})
    changes = []
    for main in [c for c in comparisons if c['role'] == 'primary']:
        check = next((c for c in comparisons if c['role'] == 'check' and c['generator'] == main['generator']
                      and c['arm'] == main['arm'] and c['reference'] == main['reference'] and c['measure'] == main['measure']), None)
        # Compare directions on the SAME presampled task blocks. A full-set
        # primary effect versus a three-task check could confound task choice.
        if check:
            primary_values = dict(zip(main['tasks_used'], main['per_task_differences']))
            check_values = dict(zip(check['tasks_used'], check['per_task_differences']))
            common = sorted(primary_values.keys() & check_values.keys())
        else:
            common = []
        if common:
            primary_difference = statistics.mean(primary_values[t] for t in common)
            check_difference = statistics.mean(check_values[t] for t in common)
            changes.append({'generator': main['generator'], 'arm': main['arm'], 'reference': main['reference'], 'measure': main['measure'],
                            'tasks_used': common, 'primary_check_tasks_difference': primary_difference, 'check_difference': check_difference,
                            'direction_reversed': primary_difference * check_difference < 0})
    guards=[]
    check_quality_guard(cfg)
    guard=cfg['quality_guard']
    for metric,margin in guard['margins'].items():
        # A deterministic constraint guard has support only on tasks with
        # applicable constraints. Do not count other tasks as lost data.
        eligible=eligible_tasks(tasks, metric)
        policy=support_policy(guard, metric)
        for contrast in paired_comparisons(rows,arms,cfg['repetitions'],eligible,'primary',metric,cfg['seed'],guard['reference'],models=models):
            guards.append(contrast|{'margin':margin,'support_policy':policy,'verdict':guard_verdict(contrast,margin,policy)}
                          | missingness_diagnostic(contrast))
    guard_note = ('Automatic exploratory safeguard on declared scores; not confirmatory non-inferiority or human-quality proof.'
                  if provenance['exploratory'] else
                  'Automatic score safeguard; this report does not infer confirmatory status or human-quality validation.')
    for guarded in guards:
        guarded['guard_interpretation'] = guard_note
    costs = request_costs(output, rows, result['accounting'])
    summaries = arm_summaries(rows, cfg, tasks, result['n'], costs, embeddings)
    missing_primary = [c for c in analysis['primary_contrasts'] if c['measure'] in EMBEDDING_FIELDS and not embeddings]
    report = {'schema': 'openrouter-main-report-v3', 'protocol': cfg['protocol'], 'comparisons': comparisons,
              **{key: provenance[key] for key in EXECUTION_FLAGS}, 'execution_provenance': provenance,
              'manifest_sha256': data.sha(output / 'manifest.json'),
              'analysis_reporting': reporting, 'secondary_task_effects': secondary,
              'arm_diagnostics': rates, 'cells': diagnostics, 'accounting': result['accounting'],
              'analysis': analysis, 'uncomputed_primary_contrasts': missing_primary,
              'arm_summaries': summaries, 'default_saturation': saturation, 'cost_allocation': costs,
              'results_sha256': data.sha(output / 'results.json'), 'evaluator_direction_checks': changes,
              'environment': environment(), 'report_code_sha256': code_hashes(),
              'quality_guards':guards,'quality_guard_protocol':guard,
              'interpretation': ('Exploratory comparison with explicitly retained failed-gate exceptions; not confirmatory. '
                  if provenance['exploratory'] else 'Execution status is unspecified; not inferred to be confirmatory. ')
                  + 'Judge-assessed differences and prespecified automatic quality-guard intervals. No human validation or held-out generalisation test. Inconclusive guards are not evidence of preserved quality.'}
    op.write_json(output / 'report.json', report)
    export_tables(output, report)
    lines = ['# API study results', '', report['interpretation'], '',
             f"Execution status: {provenance['status']}; exploratory={json.dumps(provenance['exploratory'])}; "
             f"confirmatory={json.dumps(provenance['confirmatory'])}; main_eligible={json.dumps(provenance['main_eligible'])}.",
             f"Exploratory authorization SHA-256: {provenance['authorization_sha256'] or 'not declared' }.", '',
             f"Analysis specification: {analysis['status']}; decision: {analysis['decision_reference']}.", '',
             'Arm means are descriptive on their own available task support. Use direct paired contrasts for comparisons.',
             'Quality cutoffs are operational judge-score thresholds, not calibrated human-quality boundaries.',
             'Flag-free means only that the declared empty/refusal/script/truncation checks did not fire; constraints are separate.', '',
             '| Generator | Arm | Measure | Task-weighted mean | Complete tasks |',
             '|---|---|---|---:|---:|']
    for arm in summaries:
        for field in ('distinct_count', 'quality', 'constraint_satisfaction'):
            measure = arm['metrics'][field]
            lines.append(f"| {arm['generator']} | {arm['arm']} | {field} | {measure['mean']} | {len(measure['tasks_used'])}/{len(measure['tasks_used'])+len(measure['tasks_lost'])} |")
    lines += ['', '## Fresh default saturation', '',
              'Counts use complete primary groupings in individual fresh default task/repetition cells, before repetition averaging. '
              'The three count categories sum to available cells; unavailable cells are separate. '
              'At n−1, one additional group remains possible. All declared primary contrasts retain their original task roster.', '',
              '| Generator | Responses per cell (n) | Expected cells | Available cells | Unavailable cells | Count = n | Count = n−1 | Count ≤ n−2 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for diagnostic in saturation:
        lines.append('| ' + ' | '.join(str(diagnostic[k]) for k in
            ('generator', 'sample_count', 'expected_cells', 'available_cells', 'unavailable_cells',
             'at_ceiling_cells', 'one_below_ceiling_cells', 'at_least_two_below_ceiling_cells')) + ' |')
    lines += ['', '## Paired comparisons', '',
              '| Generator | Arm | Reference | Measure | Analysis role | Mean difference | 95% paired-task interval | Tasks |',
              '|---|---|---|---|---|---:|---|---:|']
    for comparison in comparisons:
        if comparison['role'] == 'primary':
            lines.append(f"| {comparison['generator']} | {comparison['arm']} | {comparison['reference']} | {comparison['measure']} | {comparison['analysis_role']} | {comparison['mean_difference']} | {comparison['paired_task_bootstrap_95']} | {len(comparison['tasks_used'])}/{len(tasks)} |")
    if missing_primary:
        lines += ['', '**Primary embedding contrasts are uncomputed; rerun this report with --embeddings before interpreting them.**']
    if reporting['secondary_subsets']:
        lines += ['', '## Declared secondary task views', '',
                  'These subsets were declared in the frozen authorization. Selection does not establish generalization; '
                  'the full-task comparison above remains primary. Only per-task effects are shown, with no pooled subset mean or interval.', '']
        for subset in reporting['secondary_subsets']:
            lines += [f"Subset {subset['id']}: {subset['selection_basis']}", '']
        lines += ['| Subset | Generator | Arm | Reference | Measure | Task | Difference | Availability |',
                  '|---|---|---|---|---|---|---:|---|']
        for effect in secondary:
            lines.append('| ' + ' | '.join(str(effect[k]) for k in
                ('subset_id', 'generator', 'arm', 'reference', 'measure', 'task', 'difference', 'availability')) + ' |')
    lines += ['', 'Raw attempts, missing-data support, quality-filtered counts (cutoffs 4/5/6),',
              'flag diagnostics, check-judge results and cost are retained in JSON and CSV files.',
              f"Cost allocation: {costs['status']}; unresolved/unallocated attempts remain explicit in report.json.",
              'Missing more than two tasks marks a comparison as weak. Repetitions are averaged',
              'within tasks; individual responses are not independent statistical units.', '']
    lines += [guard_note, '', '| Generator | Arm | Quality reference | Guard | Margin | Tasks used/eligible | Weak | Verdict | Scope |',
              '|---|---|---|---|---:|---:|---|---|---|']
    for guard_result in guards:
        lines.append(f"| {guard_result['generator']} | {guard_result['arm']} | {guard_result['reference']} | {guard_result['measure']} | {guard_result['margin']} | {len(guard_result['tasks_used'])}/{len(guard_result['tasks_used'])+len(guard_result['tasks_lost'])} | {guard_result['weak']} | {guard_result['verdict']} | {guard_result['verdict_scope']} |")
    lines += ['', 'A permitted missingness allowance does not establish that missing tasks behave like retained tasks.',
              'quality_guards.csv includes bounded full-task point-mean sensitivity ranges; these are not confidence intervals or replacement verdicts.', '']
    if changes:
        # Same presampled task blocks for both judges; agreement is sensitivity evidence, not validation.
        check_qualified = manifest['specification']['selection']['check_qualified']
        qualification_note = ('The check judge passed grouping qualification (check_qualified=true). '
                              if check_qualified else
                              'The check judge is not qualified for grouping (check_qualified=false). ')
        lines += ['## Check-judge direction comparison', '',
                  'Primary and check-judge differences are averaged over the same presampled check tasks. '
                  + qualification_note + 'Agreement is not independent validation.', '',
                  '| Generator | Arm | Reference | Measure | Check tasks | Primary difference | Check difference | Direction reversed |',
                  '|---|---|---|---|---|---:|---:|---|']
        for change in changes:
            lines.append(f"| {change['generator']} | {change['arm']} | {change['reference']} | {change['measure']} | "
                         f"{', '.join(change['tasks_used'])} | {change['primary_check_tasks_difference']} | "
                         f"{change['check_difference']} | {change['direction_reversed']} |")
        lines.append('')
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return report
