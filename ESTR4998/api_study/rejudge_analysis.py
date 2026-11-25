"""Descriptive, development-only analysis of saved-response rejudgments.

No generation, quality, original endpoint, qualification gate or human rating is
changed. A partition labels response identities, not semantic truth. Consequently
split/merge below mean changes relative to an earlier partition, not errors.

Input cells have task_id, arm, two original/repeats records (repetition,
response_ids, labels), and one pooled record (response_ids, labels). New labels
may be None for failed/unavailable judgments. Labels are in response_ids order;
the caller must first invert the API's presentation permutation. An independent
identity alignment here permits reordered records without comparing wrong pairs.
"""
from itertools import combinations
import math
import random
import statistics

import api_evaluation as ev

SCHEMA = 'saved-response-rejudge-analysis-v1'


def _require(value, message):
    if not value:
        raise ValueError(message)


def _partition(record, *, missing=False):
    ids, labels = record['response_ids'], record['labels']
    _require(isinstance(ids, list) and len(ids) >= 2
             and all(isinstance(v, str) and v for v in ids)
             and len(set(ids)) == len(ids), 'Response IDs must be unique nonempty strings')
    if labels is None:
        _require(missing, 'Original partition cannot be missing')
        return ids, None
    _require(isinstance(labels, list) and len(labels) == len(ids)
             and all(type(v) in (int, str) for v in labels),
             'Partition requires one integer/string label per response ID')
    return ids, labels


def _aligned(record, ids, *, missing=False):
    actual, labels = _partition(record, missing=missing)
    _require(set(actual) == set(ids), 'Partition response identities differ')
    lookup = dict(zip(actual, labels)) if labels is not None else None
    return [lookup[v] for v in ids] if lookup is not None else None


def partition_change(before, after):
    """Compare unordered distinct response pairs, invariant to group names.

    A split is a previously co-grouped pair now separated; a merge is a previously
    separated pair now co-grouped. Rates condition on their respective original
    pair populations and are None when those populations are empty. Agreement
    includes both persistent same and persistent different pairs. ARI supplements
    agreement because singleton-heavy partitions can have high agreement.
    """
    _require(len(before) == len(after) and len(before) >= 2,
             'Pair comparison requires equal partitions of at least two responses')
    same = different = split = merge = 0
    for i, j in combinations(range(len(before)), 2):
        if before[i] == before[j]:
            same += 1
            split += after[i] != after[j]
        else:
            different += 1
            merge += after[i] == after[j]
    pairs = same + different
    return {'pairs': pairs, 'original_same_pairs': same,
            'original_different_pairs': different, 'split_pairs': split,
            'merged_pairs': merge, 'split_rate': split / same if same else None,
            'merge_rate': merge / different if different else None,
            'agreement': (pairs - split - merge) / pairs,
            'adjusted_rand_index': ev.ari(before, after),
            'same_partition': split == merge == 0}


def _mean(values):
    return statistics.mean(values) if values else None


def _counts(labels):
    return len(set(labels)) if labels is not None else None


def _percentile(sorted_values, probability):
    """Linearly interpolate the empirical bootstrap quantile (type 7)."""
    index = (len(sorted_values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return sorted_values[lo] + (index - lo) * (sorted_values[hi] - sorted_values[lo])


def _joint_intervals(columns, tasks, draws, seed):
    """Resample tasks, keeping all repetitions/conditions of a task together.

    Every column uses the same task draw. Each column is already an equally
    weighted task-level mean/contrast, so responses, pairs and repetitions are
    never treated as independent experimental units. Identical retained-task
    rosters share deterministic draws across contrasts; changed support is stated
    explicitly. Intervals are descriptive percentile intervals, not gate tests.
    """
    if len(tasks) < 2:
        return {key: None for key in columns}
    rng = random.Random(f'{seed}|{tasks!r}')
    samples = {key: [] for key in columns}
    for _ in range(draws):
        indices = rng.choices(range(len(tasks)), k=len(tasks))
        for key, values in columns.items():
            samples[key].append(statistics.mean(values[i] for i in indices))
    return {key: [_percentile(sorted(values), .025), _percentile(sorted(values), .975)]
            for key, values in samples.items()}


def _validate(cells):
    _require(isinstance(cells, list) and cells, 'Cells must be a nonempty list')
    lookup = {}
    for cell in cells:
        task, arm = cell['task_id'], cell['arm']
        _require(isinstance(task, str) and task and isinstance(arm, str) and arm,
                 'Task/arm identities must be nonempty strings')
        key = task, arm
        _require(key not in lookup, 'Duplicate task/arm cell')
        originals, repeats = cell['original'], cell['repeats']
        _require(isinstance(originals, list) and isinstance(repeats, list)
                 and len(originals) == len(repeats) == 2,
                 'Exactly two original and two repeat records required')
        _require(all(type(r['repetition']) is int for r in originals + repeats)
                 and sorted(r['repetition'] for r in originals) == [0, 1]
                 and sorted(r['repetition'] for r in repeats) == [0, 1],
                 'Repetitions must be exactly 0 and 1')
        originals = sorted(originals, key=lambda r: r['repetition'])
        repeats = sorted(repeats, key=lambda r: r['repetition'])
        ids, before, after = [], [], []
        for original, repeat in zip(originals, repeats):
            rid, labels = _partition(original)
            ids.append(rid)
            before.append(labels)
            after.append(_aligned(repeat, rid, missing=True))
        _require(len(ids[0]) == len(ids[1]), 'Repetitions must have equal response counts')
        pooled_ids = ids[0] + ids[1]
        _require(len(set(pooled_ids)) == len(pooled_ids),
                 'Repetitions must have disjoint response identities')
        pooled = _aligned(cell['pooled'], pooled_ids, missing=True)
        lookup[key] = {'ids': ids, 'before': before, 'after': after, 'pooled': pooled}
    tasks = sorted({key[0] for key in lookup})
    arms = sorted({key[1] for key in lookup})
    _require('default' in arms, 'Default arm is required')
    _require(len(lookup) == len(tasks) * len(arms),
             'Include the full task/arm roster; unavailable new labels must be None')
    return lookup, tasks, arms


def analyze(cells, *, bootstrap_samples=5000, seed=20261115):
    """Return transparent support, repeat stability and pooled diagnostics.

    Task-arm counts average both repetitions. A contrast retains a task only
    when both arms have both repeat judgments; original, repeat and change in
    contrast use exactly that same support. Change = repeat contrast minus
    original contrast. Pooled counts are descriptive: n/context/order all change,
    and independent old labels cannot establish cross-repetition sameness.
    """
    _require(type(bootstrap_samples) is int and bootstrap_samples >= 100,
             'At least 100 bootstrap samples required')
    lookup, tasks, arms = _validate(cells)
    rows, pooled_rows = [], []
    for (task, arm), data in sorted(lookup.items()):
        before, after = data['before'], data['after']
        n = len(before[0])
        for rep in (0, 1):
            old_count, new_count = _counts(before[rep]), _counts(after[rep])
            rows.append({'task_id': task, 'arm': arm, 'repetition': rep,
                         'n': n, 'original_count': old_count, 'repeat_count': new_count,
                         'count_change': None if new_count is None else new_count - old_count,
                         'absolute_count_change': None if new_count is None else abs(new_count - old_count),
                         'original_at_ceiling': old_count == n,
                         'repeat_at_ceiling': None if new_count is None else new_count == n,
                         'partition_change': None if after[rep] is None
                         else partition_change(before[rep], after[rep])})
        pooled = data['pooled']
        pooled_row = {'task_id': task, 'arm': arm, 'n': 2 * n,
                      'pooled_count': _counts(pooled), 'at_ceiling': None if pooled is None
                      else _counts(pooled) == 2 * n, 'within_repetition': [],
                      'cross_repetition_pairs': n * n, 'cross_repetition_same_pairs': None,
                      'cross_repetition_same_rate': None}
        for rep in (0, 1):
            subset = None if pooled is None else pooled[rep * n:(rep + 1) * n]
            pooled_row['within_repetition'].append({
                'repetition': rep, 'restricted_pooled_count': _counts(subset),
                'versus_original': None if subset is None
                else partition_change(before[rep], subset),
                'versus_repeat': None if subset is None or after[rep] is None
                else partition_change(after[rep], subset)})
        if pooled is not None:
            cross = sum(pooled[i] == pooled[j] for i in range(n) for j in range(n, 2 * n))
            pooled_row.update(cross_repetition_same_pairs=cross,
                              cross_repetition_same_rate=cross / (n * n))
        pooled_rows.append(pooled_row)
    contrasts = []
    pairs = [(arm, 'default') for arm in arms if arm != 'default']
    if 'manual_causal_shift_v1' in arms and 'naive_creative' in arms:
        pairs.append(('manual_causal_shift_v1', 'naive_creative'))
    for arm, reference in pairs:
        retained, missing = [], []
        columns = {'original': [], 'repeat': [], 'change': []}
        for task in tasks:
            a, b = lookup[task, arm], lookup[task, reference]
            if any(labels is None for labels in a['after'] + b['after']):
                missing.append(task)
                continue
            retained.append(task)
            old = statistics.mean(_counts(v) for v in a['before']) - statistics.mean(_counts(v) for v in b['before'])
            new = statistics.mean(_counts(v) for v in a['after']) - statistics.mean(_counts(v) for v in b['after'])
            columns['original'].append(old)
            columns['repeat'].append(new)
            columns['change'].append(new - old)
        intervals = _joint_intervals(columns, retained, bootstrap_samples, seed)
        contrasts.append({'arm': arm, 'reference': reference,
                          'tasks_expected': len(tasks), 'tasks_used': retained, 'tasks_missing': missing,
                          'n_tasks': len(retained), 'repetitions_per_task': 2,
                          'per_task': {key: dict(zip(retained, values)) for key, values in columns.items()},
                          'mean': {key: _mean(values) for key, values in columns.items()},
                          'paired_task_bootstrap_95': intervals})
    summaries = []
    for arm in arms:
        arm_rows = [r for r in rows if r['arm'] == arm]
        usable = [r for r in arm_rows if r['repeat_count'] is not None]
        pooled_usable = [r for r in pooled_rows if r['arm'] == arm and r['pooled_count'] is not None]
        complete_tasks = [task for task in tasks
                          if all(v is not None for v in lookup[task, arm]['after'])]
        old = [statistics.mean(_counts(v) for v in lookup[task, arm]['before']) for task in complete_tasks]
        new = [statistics.mean(_counts(v) for v in lookup[task, arm]['after']) for task in complete_tasks]
        changes = [b - a for a, b in zip(old, new)]
        summaries.append({'arm': arm, 'sets_expected': len(arm_rows), 'repeat_sets_available': len(usable),
                          'repeat_sets_missing': len(arm_rows) - len(usable),
                          'original_sets_at_ceiling_all': sum(r['original_at_ceiling'] for r in arm_rows),
                          'original_sets_at_ceiling_matched': sum(r['original_at_ceiling'] for r in usable),
                          'repeat_sets_at_ceiling_available': sum(r['repeat_at_ceiling'] for r in usable),
                          'same_partition_sets': sum(r['partition_change']['same_partition'] for r in usable),
                          'changed_count_sets': sum(r['count_change'] != 0 for r in usable),
                          'mean_absolute_count_change_available_sets': _mean([abs(r['count_change']) for r in usable]),
                          'complete_tasks': complete_tasks, 'complete_task_count': len(complete_tasks),
                          'task_mean_original_count_matched': _mean(old),
                          'task_mean_repeat_count_matched': _mean(new),
                          'task_mean_count_change': _mean(changes),
                          'count_change_task_bootstrap_95': _joint_intervals({'change': changes}, complete_tasks,
                                                                          bootstrap_samples, seed)['change'],
                          'pooled_sets_expected': len(tasks), 'pooled_sets_available': len(pooled_usable),
                          'pooled_sets_at_ceiling': sum(r['at_ceiling'] for r in pooled_usable),
                          'pooled_count_mean_available_tasks': _mean([r['pooled_count'] for r in pooled_usable])})
    return {'schema': SCHEMA, 'development_only': True, 'qualified_endpoint': False,
            'original_endpoint_changed': False, 'tasks': tasks, 'arms': arms,
            'bootstrap': {'samples': bootstrap_samples, 'seed': seed, 'unit': 'task',
                          'method': 'paired percentile, empirical linear quantiles',
                          'interpretation': 'descriptive; ten selected development tasks are not a population sample'},
            'interpretation': [
                'Repeatability is not semantic validity; original labels are a comparator, not ground truth.',
                'Pooled n=20 changes context, order and set size; it is not a replacement endpoint or a validation gate.',
                'Cross-repetition co-membership comes only from the new joint partition, never from old label equality.',
                'No new quality, constraint, generation or human-rating evidence is introduced.',
                'Missing judgments are unavailable, never zero or an imputed singleton partition.'],
            'repeat_sets': rows, 'pooled_sets': pooled_rows,
            'arm_summaries': summaries, 'contrasts': contrasts}
