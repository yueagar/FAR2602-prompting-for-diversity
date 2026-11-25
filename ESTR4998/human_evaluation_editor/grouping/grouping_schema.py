"""Validation for one anonymous response-grouping sheet. Standard library only."""

import json
import re


GROUPING_FIELDS = [
    'item', 'packet_sha256', 'task', 'responses', 'groups', 'status', 'exposure', 'note',
]
GROUPING_EDITABLE = ['groups', 'status', 'exposure', 'note']
IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', re.ASCII)
SHA256 = re.compile(r'[0-9a-f]{64}', re.ASCII)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json_loads(text):
    """Reject duplicate keys and non-JSON numbers without echoing input text."""
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result

    def constant(_value):
        raise ValueError('Invalid JSON number')

    require(isinstance(text, str), 'JSON must be text')
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise ValueError('Invalid or duplicate JSON fields') from None


def validate_responses(value):
    require(isinstance(value, list) and 2 <= len(value) <= 20,
            'A grouping set needs 2–20 responses')
    ids = []
    for response in value:
        require(isinstance(response, dict) and set(response) == {'id', 'text'},
                'Responses must contain only an anonymous ID and text')
        require(isinstance(response['id'], str) and IDENTIFIER.fullmatch(response['id']),
                'Invalid response ID')
        require(isinstance(response['text'], str), 'Response text must be text')
        ids.append(response['id'])
    require(len(set(ids)) == len(ids), 'Response IDs must be unique')
    return ids


def validate_row(row):
    """Validate the complete stored CSV row, including its editable partition.

    Group numbers are arbitrary labels: contiguity and numerical order have no
    semantic meaning. Partial drafts may omit response IDs. A completed row
    must assign every response exactly once and record prior-exposure status.
    """
    require(isinstance(row, dict) and set(row) == set(GROUPING_FIELDS),
            'Unexpected grouping sheet columns')
    require(all(isinstance(value, str) for value in row.values()),
            'Grouping sheet fields must be text')
    require(IDENTIFIER.fullmatch(row['item']) is not None, 'Invalid grouping item ID')
    require(SHA256.fullmatch(row['packet_sha256']) is not None, 'Invalid packet hash')
    require(bool(row['task'].strip()), 'The grouping task must not be empty')
    ids = validate_responses(strict_json_loads(row['responses']))
    groups = strict_json_loads(row['groups'])
    require(isinstance(groups, dict), 'Groups must map response IDs to group numbers')
    require(set(groups) <= set(ids), 'A group assignment refers to an unknown response')
    require(all(type(label) is int and 1 <= label <= len(ids) for label in groups.values()),
            'Group numbers must be integers between 1 and the response count')
    require(row['status'] in ('draft', 'complete', 'unsure'), 'Invalid grouping status')
    require(row['exposure'] in ('', 'no', 'yes', 'unsure'), 'Invalid prior-exposure answer')
    require(len(row['note']) <= 10000, 'Notes must be at most 10,000 characters')
    if row['status'] == 'complete':
        require(set(groups) == set(ids), 'Assign every response before marking the set complete')
        require(bool(row['exposure']), 'Record prior exposure before marking the set complete')
    if row['status'] == 'unsure':
        require(bool(row['note'].strip()), 'Explain what needs discussion in the note')
