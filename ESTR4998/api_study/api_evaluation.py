"""Versioned full-response judging and qualification computations.

Class labels are local to each response set. Pair rates use only pairs for
which the respective error is possible; zero denominators are reported as
None, never silently counted as successful evidence. Frozen gates must state
their zero-denominator policy. ARI is diagnostic, not the qualification key.
"""
from collections import Counter
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import api_runner as op

PROMPTS = json.loads(Path(__file__).with_name('noveltybench_prompts.json').read_text())
for _key in ('JUDGE_SYSTEM', 'SCORE_SYSTEM'):
    op.require(hashlib.sha256(PROMPTS[_key].encode()).hexdigest() == PROMPTS['sha256'][_key], 'Pinned prompt hash mismatch')
VERSION = 'estr-noveltybench-v1.1-full-response-v1'
TASK_SENSITIVE_PROFILE = 'task-sensitive-central-content-v2'
TASK_SENSITIVE_GROUPING = PROMPTS['JUDGE_SYSTEM'] + '''

Apply this criterion at the level of the substantive idea requested by the task.
Identify a specific summary of each response's central answer, then decide
whether a difference changes that answer or only elaborates it. A shared topic,
theme, moral, narrative template or writing style alone is not enough to make
two responses equivalent.

For a narrative, compare its distinctive premise, central problem, consequential
causal development and resolution. Supporting scenes, dialogue, emotional
emphasis, or the scene in which a character discovers the same solution can
differ while preserving the same central plot. Changes that alter the central
motivation, mechanism, conflict or resolution make the answers distinct.

For a fictional biography asking for a profession and greatest contribution,
compare the substantive achievement. Names, dates, locations, supporting
methods and elaborations of the legacy do not by themselves
create a new contribution. A changed method matters when it changes the central
achievement or when the task specifically asks for that method or discovery history.
Required fields can matter for completeness without defining a different idea.

When the task specifically asks for a backstory or origin, the causal history is
central: materially different causes or consequential sequences explaining how
a character became what they are make distinct answers, even if their present
traits, behaviour or role are similar.

Names, dates, setting details and other supporting particulars can still be
central when the task or the causal explanation depends on them. Judge the full
responses in their task context; neither shared structure nor an added detail
alone decides equivalence. Keep this grouping judgment separate from writing
quality: wording, formatting or length differences alone do not create another
central answer. Do not infer equivalence or difference from generation provenance.
'''


def profile(profile_id=None):
    """Choose an explicit evaluator while preserving the pinned legacy default.

    The task-sensitive profile changes only the grouping instruction. Its
    hashes describe the actual amended text; upstream_commit identifies the
    source of the unchanged base, not upstream authorship of the amendment.
    """
    if profile_id is None or profile_id == VERSION:
        return {'version': VERSION, 'prompts': PROMPTS}
    op.require(profile_id == TASK_SENSITIVE_PROFILE, 'Unknown evaluation profile')
    prompts = PROMPTS | {
        'JUDGE_SYSTEM': TASK_SENSITIVE_GROUPING,
        'adaptation': TASK_SENSITIVE_PROFILE,
        'base_profile': VERSION,
        'sha256': {
            'JUDGE_SYSTEM': hashlib.sha256(TASK_SENSITIVE_GROUPING.encode()).hexdigest(),
            'SCORE_SYSTEM': PROMPTS['sha256']['SCORE_SYSTEM'],
        },
    }
    return {'version': TASK_SENSITIVE_PROFILE, 'prompts': prompts}


GROUP_SCHEMA = {'type': 'object', 'properties': {'groups': {'type': 'array', 'minItems': 1,
                 'items': {'type': 'array', 'minItems': 1, 'items': {'type': 'integer', 'minimum': 0}}}},
                'required': ['groups'], 'additionalProperties': False}
SCORE_SCHEMA = {'type': 'object', 'properties': {'score': {'type': 'integer', 'minimum': 1, 'maximum': 10}},
                'required': ['score'], 'additionalProperties': False}


def render_responses(prompt, responses):
    """Exact wrapper from pinned upstream src/common.py (MIT)."""
    parts = [f'<prompt>\n{prompt}\n</prompt>']
    for i, response in enumerate(responses):
        parts.append(f'<response index="{i}">\n{response}\n</response>')
    return '\n\n'.join(parts)


def messages(task, responses, role, order=None, profile_id=None):
    prompts = profile(profile_id)['prompts']
    if role == 'grouping':
        order = list(range(len(responses))) if order is None else order
        op.require(sorted(order) == list(range(len(responses))), 'Invalid shuffle map')
        return [{'role': 'system', 'content': prompts['JUDGE_SYSTEM']},
                {'role': 'user', 'content': render_responses(task, [responses[i] for i in order])}]
    op.require(role == 'quality' and len(responses) == 1, 'Quality judges one response per call')
    return [{'role': 'system', 'content': prompts['SCORE_SYSTEM']},
            {'role': 'user', 'content': f'<prompt>\n{task}\n</prompt>\n\n<response>\n{responses[0]}\n</response>'}]


def judge_body(route, task, responses, role, order=None, profile_id=None):
    body = op.payload(route, messages(task, responses, role, order, profile_id), route['parameters'])
    if route['json_schema']:
        body['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': role, 'strict': True, 'schema': GROUP_SCHEMA if role == 'grouping' else SCORE_SCHEMA}}
    else:
        # Explicit, recorded adaptation. Never silently repair a malformed reply.
        body['messages'][0]['content'] += ('\nReturn JSON only, with no additional fields: '
                                         + ('{"groups": [[0, 1], [2]]}.' if role == 'grouping' else '{"score": 7}.'))
    return body


def canonical(labels):
    seen = {}
    return [seen.setdefault(value, len(seen)) for value in labels]


def strict_json(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            op.require(key not in obj, 'Repeated JSON field')
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=unique)


def parse_grouping(text, order):
    obj = strict_json(text)
    op.keys(obj, {'groups'}, 'grouping reply')
    groups = obj['groups']
    op.require(isinstance(groups, list) and groups and all(isinstance(g, list) and g for g in groups), 'Groups must be nonempty lists')
    flat = [i for group in groups for i in group]
    op.require(all(type(i) is int for i in flat) and sorted(flat) == list(range(len(order))),
               'Grouping must cover each shown index exactly once')
    labels = [None] * len(order)
    for group, indices in enumerate(groups):
        for shown in indices:
            labels[order[shown]] = group
    return canonical(labels)


def parse_quality(text):
    obj = strict_json(text)
    op.keys(obj, {'score'}, 'quality reply')
    op.require(type(obj['score']) is int and 1 <= obj['score'] <= 10, 'Score must be an integer 1–10')
    return obj['score']


def shuffle(n, seed, identity):
    order = list(range(n))
    random.Random(op.fingerprint([seed, identity])).shuffle(order)
    return order


def ari(a, b):
    op.require(len(a) == len(b) and len(a) > 0, 'ARI requires equal nonempty partitions')
    if len(a) < 2:
        return 1.0
    choose = lambda n: n * (n - 1) / 2
    overlap = sum(choose(n) for n in Counter(zip(a, b)).values())
    left, right = (sum(choose(n) for n in Counter(x).values()) for x in (a, b))
    expected = left * right / choose(len(a))
    denominator = (left + right) / 2 - expected
    return 1.0 if denominator == 0 else (overlap - expected) / denominator


def partition_errors(expected, observed):
    op.require(len(expected) == len(observed), 'Partition length mismatch')
    same = different = merged = split = 0
    for i, j in combinations(range(len(expected)), 2):
        if expected[i] == expected[j]:
            same += 1
            split += observed[i] != observed[j]
        else:
            different += 1
            merged += observed[i] == observed[j]
    return {'false_merges': merged, 'different_pairs': different, 'false_splits': split, 'same_pairs': same,
            'false_merge_rate': merged / different if different else None,
            'false_split_rate': split / same if same else None, 'ari': ari(expected, observed)}


def borderline_partition(expected, scenario):
    """Deterministic fixture perturbations, never fitted to a candidate judge.

    A merge joins two whole classes; a split moves one member to a new class.
    A label swap between two non-singleton classes preserves class counts and
    sizes while creating both false merges and false splits. Inapplicable
    patterns return the gold partition with applied=False, so an unchanged
    all-singleton/all-same fixture is not counted as a successful perturbation.
    """
    labels = canonical(expected)
    groups = {label: [i for i, value in enumerate(labels) if value == label]
              for label in dict.fromkeys(labels)}
    changed = list(labels)
    if scenario == 'single_merge':
        if len(groups) < 2:
            return labels, False
        first, second = list(groups)[:2]
        changed = [first if value == second else value for value in labels]
    elif scenario == 'single_split':
        eligible = [indices for indices in groups.values() if len(indices) > 1]
        if not eligible:
            return labels, False
        changed[eligible[0][-1]] = max(labels) + 1
    elif scenario == 'count_cancelling':
        eligible = [indices for indices in groups.values() if len(indices) > 1]
        if len(eligible) < 2:
            return labels, False
        left, right = eligible[0][-1], eligible[1][0]
        changed[left], changed[right] = changed[right], changed[left]
    else:
        raise ValueError('Unknown borderline partition scenario')
    return canonical(changed), True


def group_metrics(labels, scores):
    op.require(len(labels) == len(scores), 'Scores must cover the same response set')
    shares = [v / len(labels) for v in Counter(labels).values()] if labels else []
    result = {'distinct_count': len(shares), 'effective_count': math.exp(-sum(p * math.log(p) for p in shares)) if shares else 0,
              'quality_complete': all(s is not None for s in scores)}
    for cutoff in (4, 5, 6):
        retained = [labels[i] for i, score in enumerate(scores) if score is not None and score >= cutoff]
        result[f'quality_filtered_{cutoff}'] = {
            'distinct_count': len(set(retained)) if result['quality_complete'] else None,
            'retained': len(retained), 'attempted': len(labels), 'zero_acceptance': not retained,
            'complete': result['quality_complete']}
    return result


def grouping_summary(sets, outcomes, penalty):
    """Equal-weight sets; repeats reported separately, not extra tasks.

    Count error is evaluated across the three shuffles per set. A failed
    shuffle receives the frozen absolute-error penalty. Signed bias uses valid
    outputs only with coverage. Pair errors pool eligible pairs over valid
    shuffles; failure rate remains a separate mandatory gate.
    """
    rows, errors, signed = [], [], []
    merges = different = splits = same = failed = calls = 0
    for entry in sets:
        sid, expected = entry['set_id'], [r['family'] for r in entry['responses']]
        runs = outcomes[sid]
        op.require(len(runs) == 4, 'Three shuffles and one identical-order repeat required')
        calls += len(runs)
        failed += sum(x is None for x in runs)
        valid = [x for x in runs[:3] if x is not None]
        counts = [len(set(x)) for x in valid]
        per = []
        for labels in runs[:3]:
            if labels is None:
                per.append(penalty)
                continue
            delta = len(set(labels)) - len(set(expected))
            signed.append(delta)
            per.append(abs(delta))
            e = partition_errors(expected, labels)
            merges += e['false_merges']; different += e['different_pairs']
            splits += e['false_splits']; same += e['same_pairs']
        errors.extend(per)
        rows.append({'set_id': sid, 'expected_count': len(set(expected)), 'counts': counts,
                     'count_mae': statistics.mean(per), 'count_worst': max(per),
                     'count_spread': max(counts) - min(counts) if len(counts) == 3 else None,
                     'ari': [ari(expected, x) if x is not None else None for x in runs],
                     'order_ari': statistics.mean(ari(a, b) for a, b in combinations(valid, 2)) if len(valid) >= 2 else None,
                     'repeat_ari': ari(runs[0], runs[3]) if runs[0] is not None and runs[3] is not None else None})
    return {'sets': rows, 'calls': calls, 'failed_calls': failed, 'failure_rate': failed / calls,
            'count_mae': statistics.mean(errors), 'count_worst': max(errors),
            'signed_bias_valid': statistics.mean(signed) if signed else None,
            'valid_shuffle_share': len(signed) / len(errors),
            'false_merge_rate': merges / different if different else None,
            'false_split_rate': splits / same if same else None,
            'different_pairs': different, 'same_pairs': same,
            'max_count_spread': max((r['count_spread'] if r['count_spread'] is not None else penalty) for r in rows)}


def quality_summary(pairs, outcomes):
    categories = {}
    failed = calls = 0
    for pair in pairs:
        # Each side is scored twice. Any missing call makes the pair fail.
        a, b = outcomes[pair['pair_id']]
        op.require(len(a) == len(b) == 2, 'Two calls per side required')
        calls += 4
        failed += sum(x is None for x in a + b)
        strict = pair['category'] not in ('padding', 'instructed_10')
        row = categories.setdefault(pair['category'], {'kind': 'strict' if strict else 'no_improvement',
                                                       'wins': 0, 'ties': 0, 'losses': 0, 'pairs': 0})
        row['pairs'] += 1
        if any(x is None for x in a + b):
            row['losses'] += 1
        else:
            delta = statistics.mean(a) - statistics.mean(b)
            row['wins' if delta > 0 else 'ties' if delta == 0 else 'losses'] += 1
    passed = all(r['pairs'] == 4 and (r['wins'] >= 3 if r['kind'] == 'strict' else r['losses'] <= 1)
                 for r in categories.values()) and len(categories) == 6 and failed / calls <= .05
    return {'categories': categories, 'calls': calls, 'failed_calls': failed,
            'failure_rate': failed / calls if calls else None, 'pass': passed,
            'strict_wins': sum(r['wins'] for r in categories.values() if r['kind'] == 'strict'),
            'no_improvement_failures': sum(r['losses'] for r in categories.values() if r['kind'] != 'strict')}


GATE_KEYS = {'count_mae_max', 'count_worst_max', 'false_merge_rate_max', 'false_split_rate_max',
             'count_spread_max', 'failed_count_penalty', 'tie_band', 'ranking', 'zero_pair_denominator',
             'decision_reference'}


def check_gates(gates, n=10):
    op.keys(gates, GATE_KEYS, 'grouping gates')
    for key in GATE_KEYS - {'ranking', 'zero_pair_denominator', 'decision_reference'}:
        op.require(type(gates[key]) in (int, float) and math.isfinite(gates[key]) and gates[key] >= 0,
                   f'Freeze a finite nonnegative {key} before qualification')
    op.require(gates['failed_count_penalty'] > n - 1, 'Failed-count penalty must exceed any valid absolute count error')
    op.require(gates['ranking'] in ('count_mae', 'count_worst'), 'Freeze count_mae or count_worst ranking')
    op.require(gates['zero_pair_denominator'] in ('unavailable_fails', 'unavailable_ignored'), 'Freeze zero-denominator rule')
    op.require(isinstance(gates['decision_reference'], str) and gates['decision_reference'].strip(), 'Record frozen gate decision reference')


def grouping_pass(summary, gates):
    for field, gate in [('count_mae', 'count_mae_max'), ('count_worst', 'count_worst_max'),
                        ('false_merge_rate', 'false_merge_rate_max'), ('false_split_rate', 'false_split_rate_max'),
                        ('max_count_spread', 'count_spread_max')]:
        value = summary[field]
        if value is None:
            if gates['zero_pair_denominator'] == 'unavailable_fails':
                return False
        elif value > gates[gate]:
            return False
    return summary['failure_rate'] <= .05


def select_judges(summaries, candidates, generators, gates, preference):
    """Lower error is better; tie band is measured from the best candidate.

    The amended grouping ranking and band must be frozen explicitly. Retain
    the original quality-role tie breaks and preference as a recorded choice.
    """
    op.require(set(preference) == {j['label'] for j in candidates} and len(preference) == len(candidates), 'Preference must order every judge once')
    banned = {g['family'] for g in generators}
    available = [j for j in candidates if j['family'] not in banned]
    grouping = lambda j: summaries[j['label']]['grouping_pass']
    quality = lambda j: summaries[j['label']]['quality']['pass']
    q = lambda j: summaries[j['label']]['quality']
    rank = lambda j: summaries[j['label']]['grouping'][gates['ranking']]
    def best(pool, both=False):
        low = min(rank(j) for j in pool)
        tied = [j for j in pool if rank(j) <= low + gates['tie_band']]
        return min(tied, key=lambda j: (-q(j)['strict_wins'] if both else summaries[j['label']]['grouping']['failure_rate'], preference.index(j['label'])))
    both = [j for j in available if grouping(j) and quality(j)]
    if both:
        primary = best(both, True)
        group_judge = quality_judge = primary
    else:
        gs, qs = [j for j in available if grouping(j)], [j for j in available if quality(j)]
        if not gs or not qs:
            return {'status': 'stop', 'reason': 'No eligible grouping or quality judge'}
        group_judge = best(gs)
        quality_judge = min(qs, key=lambda j: (-q(j)['strict_wins'], q(j)['no_improvement_failures'], q(j)['failure_rate'], preference.index(j['label'])))
    used = {group_judge['family'], quality_judge['family']}
    check_pool = [j for j in available if j['family'] not in used]
    check = min(check_pool, key=lambda j: (-(2 * grouping(j) + quality(j)), rank(j), preference.index(j['label']))) if check_pool else None
    return {'status': 'selected', 'grouping': group_judge['label'], 'quality': quality_judge['label'],
            'check': check['label'] if check else None,
            'check_qualified': bool(check and grouping(check) and quality(check)),
            'preference': preference}


def headroom(counts_by_model, n):
    op.require(n in (10, 20), 'Headroom stage uses 10 or 20 responses')
    rows = {}
    for label, counts in counts_by_model.items():
        complete = len(counts) == 10 and all(type(k) is int and 1 <= k <= n for k in counts)
        rows[label] = {'complete': complete, 'counts': counts,
                       'median': statistics.median(counts) if complete else None,
                       'ceiling_share': sum(k == n for k in counts) / 10 if complete else None,
                       'gaps': [n-k if k is not None else None for k in counts],
                       'near_ceiling': sum(k is not None and n-k <= 1 for k in counts)}
        rows[label]['pass'] = complete and rows[label]['median'] <= .7*n and rows[label]['ceiling_share'] <= .2
    passed = bool(rows) and all(r['pass'] for r in rows.values())
    complete = bool(rows) and all(r['complete'] for r in rows.values())
    next_stage = 'incomplete' if not complete else 'main' if passed else 'stage_20' if n == 10 else 'revise_protocol'
    return {'n': n, 'models': rows, 'pass': passed, 'next': next_stage}
