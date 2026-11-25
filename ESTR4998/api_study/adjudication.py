"""Hash-bound semantic partitions for development qualification.

The fixture's ``family`` remains generation-outline provenance. A sidecar
defines a separate semantic partition without changing any response or outline.
Its human review status is descriptive; this module never turns a development
run into scientific eligibility or infers a missing reviewer's agreement.
"""
import copy
import json
from pathlib import Path
import re

import api_runner as op
import api_data as data

SCHEMA = 'estr-adjudicated-partitions-v1'
FIELDS = {'schema', 'fixture_sha256', 'checks_sha256', 'decision_reference',
          'scope', 'reviewers', 'rationale', 'partitions', 'count_mix'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        op.require(key not in result, 'Repeated adjudication JSON field')
        result[key] = value
    return result


def _roster(scored_sets):
    op.require(isinstance(scored_sets, list) and len(scored_sets) == 12,
               'Adjudication requires all twelve scored sets')
    roster, all_ids = {}, set()
    for entry in scored_sets:
        op.require(isinstance(entry, dict) and op.identifier(entry.get('set_id')),
                   'Invalid adjudication set ID')
        sid = entry['set_id']
        op.require(sid not in roster, 'Duplicate adjudication set ID')
        op.require('exploratory' not in str(entry.get('description', '')).lower(),
                   'Exploratory sets cannot enter adjudicated scored partitions')
        responses = entry.get('responses')
        op.require(isinstance(responses, list) and len(responses) == 10,
                   'Adjudication requires ten responses per scored set')
        ids = []
        for response in responses:
            op.require(isinstance(response, dict) and op.identifier(response.get('id')),
                       'Invalid adjudication response ID')
            rid = response['id']
            op.require(rid not in all_ids, 'Duplicate adjudication response ID')
            all_ids.add(rid)
            ids.append(rid)
        roster[sid] = ids
    return roster


def _validate(amendment, scored_sets):
    op.keys(amendment, FIELDS, 'adjudication sidecar')
    op.require(amendment['schema'] == SCHEMA, 'Unknown adjudication schema')
    op.require(amendment['scope'] == 'development',
               'Adjudicated partitions are development-only')
    for key in ('fixture_sha256', 'checks_sha256'):
        value = amendment[key]
        op.require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value),
                   f'Invalid adjudication {key}')
    for key in ('decision_reference', 'rationale'):
        op.require(isinstance(amendment[key], str) and amendment[key].strip(),
                   f'Adjudication requires {key}')
    op.keys(amendment['reviewers'], {'study_owner', 'second_contributor'}, 'adjudication reviewers')
    op.require(amendment['reviewers']['study_owner'] == 'approved',
               'the study owner must approve the development partition')
    op.require(amendment['reviewers']['second_contributor'] in ('pending', 'approved', 'disagreed'),
               'Record the second contributor review status explicitly')

    roster = _roster(scored_sets)
    partitions = amendment['partitions']
    op.keys(partitions, roster, 'adjudicated partition roster')
    for sid, ids in roster.items():
        groups = partitions[sid]
        op.require(isinstance(groups, list) and 1 <= len(groups) <= 10,
                   f'Invalid partition groups for {sid}')
        flat = []
        for group in groups:
            op.require(isinstance(group, list) and group
                       and all(isinstance(rid, str) for rid in group),
                       f'Partition groups must be nonempty response-ID lists: {sid}')
            flat.extend(group)
        op.require(len(flat) == len(set(flat)),
                   f'Partition repeats a response ID: {sid}')
        op.require(len(flat) == len(ids) and set(flat) == set(ids),
                   f'Partition must cover exactly the response IDs of {sid}')

    # The bins describe the prospective diagnostic bank, not fitted thresholds.
    # Every possible count (1..10) has exactly one bin, including empty bins.
    mix = amendment['count_mix']
    op.require(isinstance(mix, list) and mix, 'Declare the adjudicated count mix')
    coverage, total = [], 0
    counts = [len(groups) for groups in partitions.values()]
    for row in mix:
        op.keys(row, {'sets', 'min_groups', 'max_groups'}, 'adjudicated count-mix bin')
        count, lo, hi = row['sets'], row['min_groups'], row['max_groups']
        op.require(type(count) is int and count >= 0
                   and type(lo) is int and type(hi) is int and 1 <= lo <= hi <= 10,
                   'Count-mix bounds and set counts must be valid integers')
        coverage.extend(range(lo, hi + 1))
        total += count
        op.require(sum(lo <= k <= hi for k in counts) == count,
                   'Adjudicated partition count mix differs from the declared mix')
    op.require(coverage == list(range(1, 11)) and total == len(roster),
               'Count-mix bins must partition counts 1..10 and cover all scored sets')
    return amendment


def load(path, fixture_path, checks_path, scored_sets):
    """Load a complete development partition bound to the original evidence.

    ``scored_sets`` must be the unchanged scored entries from the caller's
    already validated fixture. Human content checking stays with that caller;
    hash binding and exact roster coverage here prevent substitution or omission.
    The returned object retains the supplied reviewer statuses verbatim.
    """
    amendment = json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)
    _validate(amendment, scored_sets)
    op.require(data.sha(fixture_path) == amendment['fixture_sha256'],
               'Adjudication refers to different fixture bytes')
    op.require(data.sha(checks_path) == amendment['checks_sha256'],
               'Adjudication refers to different checking evidence')
    fixture = json.loads(Path(fixture_path).read_text(), object_pairs_hook=_unique_object)
    op.require(isinstance(fixture, dict) and isinstance(fixture.get('grouping_sets'), list),
               'Bound fixture must contain grouping sets')
    op.require(all(isinstance(entry, dict) for entry in fixture['grouping_sets']),
               'Invalid grouping entry in bound fixture')
    source_sets = [entry for entry in fixture['grouping_sets']
                   if 'exploratory' not in str(entry.get('description', '')).lower()]
    op.require(source_sets == scored_sets,
               'Scored sets differ from the hash-bound original fixture')
    return amendment


def scoring_sets(scored_sets, amendment):
    """Return isolated metric inputs; never export these as source fixtures.

    The old scorer consumes ``family`` as its expected class label. Replace that
    field only in a deep copy, keeping texts, source links and outline provenance
    in the original fixture unchanged. Group labels are local to each set; their
    spelling has no effect on counts, pair errors or adjusted Rand indices.
    """
    _validate(amendment, scored_sets)
    result = copy.deepcopy(scored_sets)
    for entry in result:
        labels = {rid: f'adjudicated-{index}'
                  for index, group in enumerate(amendment['partitions'][entry['set_id']])
                  for rid in group}
        for response in entry['responses']:
            response['family'] = labels[response['id']]
    return result
