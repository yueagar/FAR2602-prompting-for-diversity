"""Prospective, development-only temperature × instruction analysis.

The experimental unit is the task. For each task and condition, average the
two independent generation repetitions. Within each repetition the primary
endpoint is (K_judge1 + K_judge2)/2: a mean count, NOT a consensus partition.
No judgment is selected for being more favorable, and no missing slot is
imputed. Two judgments of the same texts do not create two experimental units.

The caller supplies the frozen plan and the full result roster, including
unavailable cells. This module is offline: it never generates, judges, downloads
an embedding model, reads credentials, or changes any original analysis.
"""
import copy
from functools import lru_cache
import hashlib
import math
import random
import statistics

import api_data as data
from rejudge_analysis import partition_change

SCHEMA = 'temperature-instruction-analysis-v1'
EMBEDDING_FIELDS = ('partition_count_proxy', 'vendi_score', 'rke_score',
                    'semantic_diversity')
INSTRUCTIONS = ('default', 'naive_creative', 'manual_causal_shift_v1')
TEMPERATURES = (.8, 1., 1.2)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mean(values):
    return statistics.mean(values) if values else None


def analysis_protocol(arms):
    """Return the complete prospective contrast family for the plan to freeze.

    Positive differences favor the first condition in a named comparison.
    Interactions are differences of instruction effects relative to T=.8.
    Every estimate has a nominal descriptive 95% interval. The five primary
    count contrasts additionally have 99% intervals (Bonferroni-style alpha/5).
    Bootstrap coverage is approximate, not a guaranteed simultaneous confidence
    statement or automatic evidence that an instruction wins.
    """
    lookup = {(a['instruction_id'], a['temperature']): a['id'] for a in arms}
    _require(len(lookup) == len(arms) == 9 and set(lookup) ==
             {(i, t) for i in INSTRUCTIONS for t in TEMPERATURES},
             'Freeze the complete three-instruction by three-temperature design')
    contrasts = []

    def add(name, role, terms):
        contrasts.append({'id': name, 'analysis_role': role,
                          'weights': {lookup[key]: weight for key, weight in terms}})

    for t in TEMPERATURES:
        add(f'naive_minus_default_at_{t:g}', 'primary',
            [(('naive_creative', t), 1), (('default', t), -1)])
    for t in TEMPERATURES[1:]:
        add(f'naive_at_0.8_minus_default_at_{t:g}', 'primary',
            [(('naive_creative', .8), 1), (('default', t), -1)])
    for t in TEMPERATURES:
        for reference, label in [('default', 'default'), ('naive_creative', 'naive')]:
            add(f'causal_minus_{label}_at_{t:g}', 'secondary',
                [(('manual_causal_shift_v1', t), 1), ((reference, t), -1)])
    for instruction, label in [('naive_creative', 'naive'),
                               ('manual_causal_shift_v1', 'causal')]:
        for t in TEMPERATURES[1:]:
            add(f'{label}_instruction_effect_at_{t:g}_minus_at_0.8', 'secondary',
                [((instruction, t), 1), (('default', t), -1),
                 ((instruction, .8), -1), (('default', .8), 1)])
    return {'version': SCHEMA, 'primary_estimator': 'mean_of_two_group_counts',
            'generation_repetitions': 2, 'primary_grouping_repetitions': 2,
            'grouping_observation_labels': 'preassigned_identifiers_not_chronological_order',
            'bootstrap_samples': 5000, 'bootstrap_seed': 20261116,
            'interval_scope': 'descriptive_unadjusted_pointwise_95_percent',
            'primary_family_interval': 'approximate_bonferroni_5_pointwise_99_percent_bootstrap',
            'contrasts': contrasts,
            'constraint_primary': 'v0.2/paired-emphasis-sentence-sensitivity',
            'constraint_sensitivity': 'frozen_original_parser',
            'embedding_secondary_fields': list(EMBEDDING_FIELDS),
            'embedding': dict(data.EMBEDDING),
            'length_diagnostic': 'whitespace_separated_words_len_text_split',
            'quality_guard': {
                'quality': {'margin': .5, 'minimum_tasks': 10, 'maximum_missing_tasks': 0},
                'constraint_satisfaction': {'margin': .05, 'minimum_tasks': 6,
                                            'maximum_missing_tasks': 0}},
            'qualified_endpoint': False, 'confirmatory': False}


@lru_cache(maxsize=32)
def _draws(tasks, count, seed):
    """Reuse identical task resamples for every contrast on identical support."""
    rng = random.Random(f'{seed}|{tasks!r}')
    return tuple(tuple(rng.choices(range(len(tasks)), k=len(tasks)))
                 for _ in range(count))


def _interval(values, tasks, count, seed, tail=.025):
    if len(values) < 2:
        return None
    samples = sorted(sum(values[i] for i in draw) / len(values)
                     for draw in _draws(tuple(tasks), count, seed))

    def percentile(p):
        position = (len(samples) - 1) * p
        lo, hi = math.floor(position), math.ceil(position)
        return samples[lo] + (position - lo) * (samples[hi] - samples[lo])

    return [percentile(tail), percentile(1 - tail)]


def with_embedding_metrics(results, embedder):
    """Return a separate copy enriched by an already instantiated local backend.

    The caller must supply the canonical pinned, full-response-window Embedder.
    This function never instantiates/downloads a model. It computes every
    complete condition symmetrically and never selects a metric from outcomes.
    Reference-relative novelty remains unavailable: no reference pool is used.
    """
    _, canonical = data.local_modules()
    _require(isinstance(embedder, canonical.Embedder) and
             all(getattr(embedder, field, None) == data.EMBEDDING[field]
                 for field in ('model_id', 'revision', 'window')),
             'Supply the already loaded canonical pinned full-response Embedder')
    enriched = copy.deepcopy(results)
    for row in enriched['rows']:
        _require('embedding_metrics' not in row, 'Do not overwrite a saved embedding result')
        if row['generation_complete']:
            texts = [g['text'] for g in row['generations']]
            row['embedding_metrics'] = data.embedding_metrics(texts, embedder) | {
                'source_text_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in texts]}
    return enriched


def response_diagnostics(task, generation):
    """Apply the same prospectively selected parser to every successful output.

    Frozen-parser results remain an explicitly separate sensitivity. A missing
    or failed response is not a failed constraint check: its missingness is
    exposed separately and excludes its task-condition from complete estimates.
    Empty successful responses remain visible to the empty-output diagnostic.
    """
    if generation is None or generation.get('status') != 'ok':
        return None
    text = generation['text']
    diagnostic = data.text_diagnostics(task, text, generation.get('finish_reason'))
    # local_modules installs the local package path; no model is instantiated.
    from localstudy import constraints_markdown_v2 as corrected
    checks = corrected.check_prompt(task, text)
    return {'flags': diagnostic['flags'], 'truncated': diagnostic['truncated'],
            'words': len(text.split()),
            'constraint_satisfied': corrected.satisfied(checks),
            'constraint_checks': checks,
            'constraint_version': corrected.CONSTRAINTS_VERSION,
            'original_constraint_satisfied': diagnostic['constraint_satisfied'],
            'original_constraint_version': diagnostic['constraint_version'],
            'flag_version': diagnostic['flag_version']}


def _labels(value, n):
    _require(value is None or isinstance(value, list) and len(value) == n and
             all(type(label) in (int, str) for label in value),
             'A grouping must label every canonical sample, or be None')
    return value


def _generation_ready(generation):
    # The runner retains an otherwise successful response with unknown billing
    # but stops before allowing that slot into downstream scientific requests.
    return generation is not None and generation.get('status') == 'ok' and generation.get('reported_cost_usd') is not None


def _validate(plan, results):
    tasks = sorted(t['id'] for t in plan['tasks'])
    arms = {a['id']: a for a in plan['arms']}
    _require(tasks and len(set(tasks)) == len(tasks), 'Task roster must be unique')
    _require(len(arms) == len(plan['arms']), 'Condition IDs must be unique')
    _require(type(plan['n']) is int and plan['n'] >= 2 and
             plan['repetitions'] == 2, 'Require n>=2 and two generation repetitions')
    _require(set(plan['check_tasks']) <= set(tasks) and
             len(set(plan['check_tasks'])) == len(plan['check_tasks']),
             'Check tasks must be a unique subset of the declared development tasks')
    protocol = plan['analysis']
    _require(protocol == analysis_protocol(plan['arms']),
             'Frozen analysis differs from the versioned prospective protocol')
    lookup = {}
    for row in results['rows']:
        c = row['cell']
        key = c['task'], c['arm'], c['repetition']
        _require(c.get('generator') == 'anchor' and c['task'] in tasks and
                 c['arm'] in arms and type(c['repetition']) is int and
                 c['repetition'] in (0, 1), 'Unexpected result cell')
        _require(key not in lookup, 'Duplicate result cell')
        condition = arms[c['arm']]
        _require(row['condition'] == {k: condition[k] for k in
                                     ('instruction_id', 'temperature')},
                 'Result condition does not match frozen plan')
        _require(row['samples'] == plan['n'] and
                 len(row['generations']) == plan['n'], 'Generation slots differ from plan')
        for generation in row['generations']:
            _require(generation is None or isinstance(generation, dict),
                     'Generation slot must contain final attempt or None')
            if generation is not None and generation.get('status') == 'ok':
                _require(isinstance(generation.get('text'), str),
                         'Successful generation requires its actual text')
        complete = all(_generation_ready(g) for g in row['generations'])
        _require(type(row['generation_complete']) is bool and
                 row['generation_complete'] == complete,
                 'generation_complete contradicts final attempt slots')
        expected_roles = {'primary', 'check'} if c['task'] in plan['check_tasks'] else {'primary'}
        _require(set(row['judges']) == expected_roles, 'Unexpected judge roster')
        for role, judge in row['judges'].items():
            count = 2 if role == 'primary' else 1
            _require(len(judge['groupings']) == count and len(judge['scores']) == plan['n'],
                     'Preserve every planned grouping and quality slot')
            for labels in judge['groupings']:
                _labels(labels, plan['n'])
                _require(labels is None or complete,
                         'Grouping cannot be present for incomplete generation')
            _require(all(s is None or type(s) is int and 1 <= s <= 10
                         for s in judge['scores']), 'Quality must be integer 1–10 or None')
            _require(all(s is None or _generation_ready(row['generations'][i])
                         for i, s in enumerate(judge['scores'])),
                     'Quality score cannot label an unavailable response')
        lookup[key] = row
    expected = {(task, arm, rep) for task in tasks for arm in arms for rep in (0, 1)}
    _require(set(lookup) == expected,
             'Include the full planned roster; unavailable slots must remain None')
    return tasks, arms, lookup


def _cell_metrics(row, diagnostics):
    values = {}
    for role, judge in row['judges'].items():
        counts = [len(set(labels)) if labels is not None else None
                  for labels in judge['groupings']]
        values[f'{role}_count'] = _mean(counts) if all(k is not None for k in counts) else None
        if role == 'primary':
            values['first_judgment_count'], values['second_judgment_count'] = counts
        scores = judge['scores']
        values[f'{role}_quality'] = _mean(scores) if all(s is not None for s in scores) else None
    for measure in ('count', 'quality'):
        primary, check = values.get(f'primary_{measure}'), values.get(f'check_{measure}')
        values[f'check_minus_primary_{measure}'] = (check - primary
            if primary is not None and check is not None else None)
    for name in ('constraint_satisfied', 'original_constraint_satisfied'):
        checks = [d[name] if d is not None else None for d in diagnostics]
        values[name] = (_mean(checks) if row['generation_complete'] and
                        all(v is not None for v in checks) else None)
    saved_embedding = row.get('embedding_metrics')
    if saved_embedding is not None:
        _, canonical = data.local_modules()
        _require(saved_embedding['embedding'] == data.EMBEDDING and
                 saved_embedding['metric_versions'] == canonical.METRIC_VERSIONS,
                 'Saved embedding metrics differ from pinned backend or versions')
        _require(row['generation_complete'] and saved_embedding['source_text_sha256'] ==
                 [hashlib.sha256(g['text'].encode()).hexdigest() for g in row['generations']],
                 'Saved embedding metrics are not bound to these ordered response texts')
    for field in EMBEDDING_FIELDS:
        value = saved_embedding['values'].get(field) if saved_embedding is not None else None
        _require(value is None or type(value) in (int, float) and math.isfinite(value),
                 'Embedding measures must be finite numbers or unavailable')
        _require(value is None or row['generation_complete'],
                 'Set embedding measures require all generated outputs')
        values[field] = value
    return values


def _estimate(per_task, eligible, protocol):
    used = sorted(per_task)
    values = [per_task[t] for t in used]
    return {'tasks_used': used, 'tasks_missing': sorted(set(eligible) - set(used)),
            'n_tasks': len(used), 'per_task': {t: per_task[t] for t in used},
            'mean': _mean(values),
            'paired_task_bootstrap_95': _interval(values, used,
                protocol['bootstrap_samples'], protocol['bootstrap_seed'])}


def _guard(estimate, policy, extent):
    """A descriptive automatic safeguard, never a validated human-quality test.

    Higher scores/rates are better. The lower interval endpoint must be at
    least -margin (inclusive). Bounds for missing tasks use the measure's
    possible contrast range; they bound a point mean, not uncertainty intervals.
    """
    bounds = estimate['paired_task_bootstrap_95']
    missing, used = len(estimate['tasks_missing']), estimate['n_tasks']
    if used < policy['minimum_tasks'] or missing > policy['maximum_missing_tasks'] or bounds is None:
        verdict = 'insufficient_support'
    elif bounds[0] >= -policy['margin'] or math.isclose(bounds[0], -policy['margin'], abs_tol=1e-12, rel_tol=0):
        verdict = 'passes_automatic_safeguard'
    elif bounds[1] < -policy['margin'] and not math.isclose(bounds[1], -policy['margin'], abs_tol=1e-12, rel_tol=0):
        verdict = 'below_margin'
    else:
        verdict = 'inconclusive'
    total, observed = used + missing, sum(estimate['per_task'].values())
    return {'policy': policy, 'verdict': verdict,
            'verdict_scope': 'complete_retained_tasks',
            'full_eligible_task_point_mean_bounds':
                [(observed - missing * extent) / total, (observed + missing * extent) / total]
                if total else None,
            'interpretation': 'Automatic exploratory safeguard only; missing-task bounds are not confidence intervals.'}


def analyze(plan, results):
    """Analyze the frozen full roster with separate support for every endpoint.

    A primary count task requires both groupings in BOTH generation repetitions
    for every condition in its contrast. First-only/second-only sensitivities
    use their own honestly reported complete support; an additional matched
    sensitivity uses exactly the primary estimator's retained tasks. Quality
    and constraints require all n outputs in both generation repetitions, and
    have their own support. No effect is divided by the observed response count.
    """
    tasks, arms, lookup = _validate(plan, results)
    protocol = plan['analysis']
    frozen_constraints, _ = data.local_modules()
    constraint_tasks = [t for t in tasks if t in frozen_constraints.PROMPT_CONSTRAINTS]
    metrics, cells = {}, []
    for key, row in sorted(lookup.items()):
        diagnostics = [response_diagnostics(key[0], g) for g in row['generations']]
        metrics[key] = _cell_metrics(row, diagnostics)
        first, second = row['judges']['primary']['groupings']
        change = partition_change(first, second) if first is not None and second is not None else None
        counts = [len(set(labels)) if labels is not None else None for labels in (first, second)]
        check = row['judges'].get('check', {}).get('groupings', [None])[0]
        cells.append({'task': key[0], 'condition': key[1], 'repetition': key[2],
                      'primary_counts': counts, 'primary_mean_count': metrics[key]['primary_count'],
                      'primary_at_ceiling': [k == plan['n'] if k is not None else None for k in counts],
                      'count_difference_second_minus_first': counts[1] - counts[0] if change is not None else None,
                      'absolute_count_difference': abs(counts[1] - counts[0]) if change is not None else None,
                      'repeat_partition_change': change,
                      'check_versus_each_primary_partition': [
                          partition_change(labels, check) if labels is not None and check is not None else None
                          for labels in (first, second)],
                      'metrics': metrics[key], 'response_diagnostics': diagnostics})
    fields = ('primary_count', 'first_judgment_count', 'second_judgment_count',
              'primary_quality', 'constraint_satisfied', 'original_constraint_satisfied',
              'check_count', 'check_quality', 'check_minus_primary_count',
              'check_minus_primary_quality') + EMBEDDING_FIELDS

    def eligible(field):
        if 'constraint' in field:
            return constraint_tasks
        if field.startswith('check_'):
            return sorted(plan['check_tasks'])
        return tasks

    task_values = {}
    for field in fields:
        for task in eligible(field):
            for arm in arms:
                values = [metrics[(task, arm, rep)].get(field) for rep in (0, 1)]
                if all(v is not None for v in values):
                    task_values[(field, task, arm)] = statistics.mean(values)
    contrasts = []
    for contrast in protocol['contrasts']:
        estimates = {}
        for field in fields:
            per_task = {}
            for task in eligible(field):
                weights = contrast['weights']
                if all((field, task, arm) in task_values for arm in weights):
                    per_task[task] = sum(weight * task_values[(field, task, arm)]
                                         for arm, weight in weights.items())
            estimates[field] = _estimate(per_task, eligible(field), protocol)
        if contrast['analysis_role'] == 'primary':
            estimate = estimates['primary_count']
            estimate['approximate_family_bootstrap_99'] = _interval(
                list(estimate['per_task'].values()), estimate['tasks_used'],
                protocol['bootstrap_samples'], protocol['bootstrap_seed'], .005)
            estimate['full_task_support_for_primary_interpretation'] = (
                len(tasks) == 10 and not estimate['tasks_missing'])
        support = estimates['primary_count']['tasks_used']
        matched = {field: _estimate({t: estimates[field]['per_task'][t] for t in support},
                                    tasks, protocol)
                   for field in ('first_judgment_count', 'second_judgment_count')}
        # Guards apply to simple arm differences only, not interactions.
        guards = {}
        if sorted(contrast['weights'].values()) == [-1, 1]:
            for name, field, extent in [('quality', 'primary_quality', 9),
                                        ('constraint_satisfaction', 'constraint_satisfied', 1)]:
                guards[name] = _guard(estimates[field], protocol['quality_guard'][name], extent)
        contrasts.append(contrast | {'estimates': estimates,
                         'single_judgment_sensitivities_on_primary_support': matched,
                         'automatic_safeguards': guards})
    summaries = []
    for arm, condition in sorted(arms.items()):
        selected = [c for c in cells if c['condition'] == arm]
        observed = [d for c in selected for d in c['response_diagnostics'] if d is not None]
        changes = [c['repeat_partition_change'] for c in selected if c['repeat_partition_change'] is not None]
        generation_slots = [g for (task, a, rep), r in lookup.items() if a == arm for g in r['generations']]
        summary_metrics = {field: _estimate({t: task_values[(field, t, arm)] for t in eligible(field)
                                            if (field, t, arm) in task_values}, eligible(field), protocol)
                           for field in fields}
        flag_names = sorted({flag for d in observed for flag in d['flags']})
        summaries.append({'condition': arm, 'instruction_id': condition['instruction_id'],
            'temperature': condition['temperature'], 'estimates': summary_metrics,
            'planned_cells': len(selected),
            'available_groupings_by_judgment': [sum(c['primary_counts'][j] is not None for c in selected) for j in (0, 1)],
            'ceiling_cells_by_judgment': [sum(c['primary_at_ceiling'][j] is True for c in selected) for j in (0, 1)],
            'repeat_comparable_cells': len(changes),
            'same_partition_cells': sum(c['same_partition'] for c in changes),
            'same_count_cells': sum(c['absolute_count_difference'] == 0 for c in selected),
            'changed_pairs': sum(c['split_pairs'] + c['merged_pairs'] for c in changes),
            'comparable_pairs': sum(c['pairs'] for c in changes),
            'mean_absolute_count_change': _mean([c['absolute_count_difference'] for c in selected
                                               if c['absolute_count_difference'] is not None]),
            'generation_diagnostics': {
                'planned_slots': len(generation_slots), 'successful_outputs': len(observed),
                'failed_final_attempts': sum(g is not None and g.get('status') != 'ok' for g in generation_slots),
                'unobserved_slots': sum(g is None for g in generation_slots),
                'successful_texts_with_unresolved_billing': sum(
                    g is not None and g.get('status') == 'ok' and g.get('reported_cost_usd') is None
                    for g in generation_slots),
                'flags': {f: sum(bool(d['flags'].get(f)) for d in observed) for f in flag_names},
                'truncated_outputs': sum(d['truncated'] for d in observed),
                'mean_words_successful_outputs': _mean([d['words'] for d in observed]),
                'constraint_applicable_outputs': sum(d['constraint_satisfied'] is not None for d in observed),
                'constraint_passed_outputs': sum(d['constraint_satisfied'] is True for d in observed),
                'original_constraint_passed_outputs': sum(d['original_constraint_satisfied'] is True for d in observed)}})
    return {'schema': SCHEMA, 'development_only': True, 'exploratory': True,
            'confirmatory': False, 'qualified_endpoint': False,
            'original_failed_gates_changed': False, 'analysis_protocol': protocol,
            'n': plan['n'], 'tasks': tasks, 'constraint_tasks': constraint_tasks,
            'check_tasks': sorted(plan['check_tasks']), 'planned_cells': len(lookup),
            'cells': cells, 'condition_summaries': summaries, 'contrasts': contrasts,
            'accounting': results.get('accounting'),
            'interpretation': 'Mean judge counts are not consensus partitions or semantic validity. '
                'The nominal 95% task-bootstrap intervals are descriptive, pointwise, and unadjusted for multiplicity. '
                'The five primary counts also have 99% Bonferroni-style bootstrap intervals, '
                'an approximate family adjustment, not guaranteed coverage. '
                'First/second judgment means preassigned observation index, not chronological order or temporal drift. '
                'All first/second sensitivities are reported; none selects a winner. '
                'Quality, constraints, failure diagnostics, embeddings, and cost remain separate.'}
