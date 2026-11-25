"""Durable, resumable API calls with accounting across every attempt.

A known-cost invalid judgment is data, not a reason to stop qualification.
Unknown billing, setup errors and exceeded reserves stop new submissions.
Generation is never retried to obtain a different answer. A no-reply transport
attempt has one technical retry after billing is known, excluding setup errors.
Judgment retry limits
are supplied by the phase: zero in qualification, one in headroom/main.
"""
from contextlib import contextmanager
import fcntl
import json
import math
import platform
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import subprocess
import time

import api_runner as op
import api_data as data
import runner_foundation_v1 as foundation
from research_client import diagnostic_context

HERE = Path(__file__).resolve().parent


def no_reply_retryable(attempt):
    """Historical retry rule: a billed attempt with no observed reply.

    An empty string is still an observed output. Setup errors remain stops,
    including older records whose HTTP status was only saved in error text.
    This predicate does not grant retries; callers enforce the declared cap.
    """
    import re
    setup = attempt.get('setup_error') or re.search(r'\bHTTP (400|401|402|403|404|405|422)\b', attempt.get('error',''))
    return (attempt.get('status') in ('failed','started') and attempt.get('response') is None
            and attempt.get('text') is None and attempt.get('reported_cost_usd') is not None and not setup)


def environment():
    packages = {}
    for name in ('numpy','transformers','torch','sentencepiece'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages}


def code_hashes():
    paths = list(HERE.glob('*.py')) + [HERE / 'noveltybench_prompts.json', HERE.parent / 'human_evaluation_editor/fixtures/fixture_sheets.py']
    paths += list((HERE.parent / 'local_model_study/src/localstudy').glob('*.py'))
    return {str(p.relative_to(HERE.parent)): op.hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


@contextmanager
def run_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process is using this run directory') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def bind_run(output, specification):
    """Bind exact inputs and implementation before any transmission."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    hashes = code_hashes()
    if path.exists():
        old = json.loads(path.read_text())
        op.require(old['specification'] == specification, 'Run inputs changed; use a new directory')
        op.require(old['code_sha256'] == hashes, 'Implementation changed since this run; use a new directory')
        return old
    op.require(not (output / 'requests').exists(), 'Unbound request directory; preserve it and use a new directory')
    try:
        commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(HERE), 'status', '--porcelain'], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    result = {'schema': 'openrouter-research-run-v1', 'created_utc': op.utcnow(),
              'specification': specification, 'code_sha256': hashes,
              'git_commit': commit, 'git_dirty': dirty, 'environment': environment()}
    op.write_json(path, result)
    return result


class Journal:
    def __init__(self, output, client, limits, base=HERE):
        self.output, self.client, self.limits = Path(output), client, limits
        self.base = Path(base)
        manifest_path = self.output / 'manifest.json'
        snapshots = (json.loads(manifest_path.read_text())['specification'].get('snapshots', {})
                     if manifest_path.exists() else {})
        self.snapshot_hashes = {entry['sha256'] for entry in snapshots.values() if entry}
        self.requests = self.output / 'requests'
        self.requests.mkdir(parents=True, exist_ok=True)
        # The orchestrator holds run_lock for this object's lifetime. Read once
        # instead of parsing the entire growing journal before every API call.
        self._records = {p.stem: json.loads(p.read_text()) for p in sorted(self.requests.glob('*.json'))}

    def records(self):
        return list(self._records.values())

    def account(self):
        attempts = [a for r in self.records() for a in r['attempts']]
        return {'attempted_requests': len(attempts),
                'known_cost_usd': sum(a.get('reported_cost_usd') or 0 for a in attempts),
                'unknown_cost_attempts': sum(a.get('reported_cost_usd') is None for a in attempts),
                'committed_usd': sum(a.get('reported_cost_usd') if a.get('reported_cost_usd') is not None
                                     else a['reserve_usd'] for a in attempts)}

    def check_budget(self, reserve):
        attempts = [a for r in self.records() for a in r['attempts']]
        op.require(not any(a.get('routing_mismatch') for a in attempts),
                   'Returned route mismatch; scheduling stopped')
        totals = self.account()
        op.require(not totals['unknown_cost_attempts'], 'Unknown billing: reconcile the saved attempt before continuing')
        op.require(not any(a['reported_cost_usd'] > a['reserve_usd'] for a in attempts),
                   'Saved request exceeded its reserve; scheduling stopped')
        op.require(not any(a.get('setup_error') and not a.get('setup_resolution') for a in attempts),
                   'Setup error: resolve configuration before continuing')
        op.require(totals['attempted_requests'] < self.limits['max_requests'], 'Request budget reached')
        op.require(totals['committed_usd'] + reserve <= self.limits['max_cost_usd'] + 1e-12,
                   'Cost budget would be exceeded')

    def call(self, call_id, route, body, metadata, parse=None, retries=0,
             *, generation_attempt_limit=2):
        op.require(op.identifier(call_id), 'Invalid request identifier')
        op.require(type(generation_attempt_limit) is int and generation_attempt_limit in (2, 3),
                   'Unsupported generation attempt limit')
        parameters = {k: v for k, v in body.items() if k in op.PARAMETERS}
        saved_route = data.runtime_route(route, parameters, self.base)
        op.require(not self.snapshot_hashes or saved_route['snapshot_sha256'] in self.snapshot_hashes,
                   'Endpoint snapshot changed since the run was bound')
        input_upper, priced_reserve = foundation._reservation(body, saved_route)
        op.require(input_upper + body['max_tokens'] <= saved_route['context_length'],
                   'Request exceeds conservative context ceiling')
        # Keep the declared reserve as a floor, never as a substitute for pricing.
        # Round the Decimal bound upwards when converting to the journal's float.
        reserve = max(route['reserve_usd'], math.nextafter(float(priced_reserve), math.inf))
        path = self.requests / (call_id + '.json')
        record = self._records.get(call_id) or {
            'id': call_id, 'request': body, 'metadata': metadata, 'attempts': []}
        self._records[call_id] = record
        op.require(record['request'] == body and record['metadata'] == metadata, 'Saved request differs from run inputs')
        attempts = record['attempts']
        generation = metadata.get('role') == 'generation'
        if generation_attempt_limit == 3:
            # A separately frozen, request-specific screen amendment grants
            # this one extra attempt only after the ordinary two have failed
            # without an observed reply. It cannot expand a fresh request.
            op.require(generation and len(attempts) in (2, 3)
                       and all(no_reply_retryable(a) for a in attempts[:2]),
                       'Extra generation attempt requires two known no-reply failures')

        def validate_route(attempt):
            raw = attempt['response']
            choices = raw.get('choices') if isinstance(raw, dict) else None
            errors = ([raw.get('error')] if isinstance(raw, dict) and raw.get('error') else [])
            errors += [c['error'] for c in choices if isinstance(c, dict) and c.get('error')] if isinstance(choices, list) else []
            if errors:
                codes = [e.get('code') for e in errors if isinstance(e, dict)]
                status = next((c for c in codes if type(c) is int and 400 <= c <= 599), None)
                attempt['failure_diagnostic'] = {'category': 'api_error', 'http_status': status}
                attempt['setup_error'] = status in (400, 401, 402, 403, 404, 405, 422)
                raise ValueError('API error; provider details suppressed')
            try:
                attempt['routing_evidence'] = foundation.validate_returned_route(raw, route, saved_route)
                attempt['routing_evidence'].update(canonical_slug=saved_route['canonical_slug'],
                                                   canonical_slug_source=saved_route['canonical_slug_source'])
            except ValueError:
                attempt.update(routing_mismatch=True, status='failed',
                               failure_diagnostic={'category': 'routing_mismatch', 'http_status': None})
                op.write_json(path, record)
                raise ValueError('Returned route mismatch; scheduling stopped') from None

        for a in attempts:
            op.require(not a.get('routing_mismatch'), 'Returned route mismatch; scheduling stopped')
            op.require(a.get('reported_cost_usd') is not None, 'Unknown billing: reconcile the saved attempt')
            op.require(a['reported_cost_usd'] <= a['reserve_usd'], 'Saved request exceeded its reserve; revise budget')
            op.require(not a.get('setup_error') or a.get('setup_resolution'), 'Setup error: resolve configuration before continuing')
        if attempts and attempts[-1]['status'] == 'started' and 'response' in attempts[-1]:
            # A process can stop after the raw response was persisted but before
            # local parsing finished. Recover locally instead of billing again.
            attempt = attempts[-1]
            stage = 'route_validation'
            try:
                validate_route(attempt)
                stage = 'response_validation'
                text, finish = op.response_text(attempt['response'], generation=generation)
                attempt.update(text=text, finish_reason=finish)
                if parse is not None:
                    op.require(finish == 'stop', 'Truncated judge response')
                    stage = 'judgment_parse'
                    attempt['parsed'] = parse(text)
                attempt['status'] = 'ok'
            except ValueError as exc:
                diagnostic = diagnostic_context(exc, attempt, stage, attempt.get('failure_diagnostic'))
                attempt.update(status='failed', error='Saved response validation failed; details suppressed',
                               failure_diagnostic=diagnostic,
                               setup_error=attempt.get('setup_error', False) or diagnostic['http_status'] in (400,401,402,403,404,405,422))
            attempt['recovered_utc'] = op.utcnow()
            op.write_json(path, record)
            if attempt.get('failure_diagnostic'):
                op.write_json(self.output / 'last_failure.json', {
                    'request_id': call_id, 'attempt': attempt['attempt'],
                    'failure_diagnostic': attempt['failure_diagnostic'], 'accounting': self.account()})
        op.require(not any(a.get('routing_mismatch') for r in self.records() for a in r['attempts']),
                   'Returned route mismatch; scheduling stopped')
        if attempts and attempts[-1]['status'] == 'ok':
            validate_route(attempts[-1])
            return attempts[-1]
        # Judge retries follow the phase limit. Generations have one narrowly
        # defined no-reply retry, independent of the judge retry setting.
        while (not attempts or (len(attempts) < generation_attempt_limit and no_reply_retryable(attempts[-1]) if generation
                                else len(attempts) <= retries)):
            # Optional client hook: binding checks run before an attempt is recorded,
            # so a stop here leaves no started/unknown-billing attempt behind.
            hook = getattr(self.client, 'before_attempt', None)
            if hook is not None:
                hook()
            self.check_budget(reserve)
            attempt = {'attempt': len(attempts), 'status': 'started', 'started_utc': op.utcnow(),
                       'reserve_usd': reserve, 'reported_cost_usd': None,
                       'priced_reserve_decimal_usd': str(priced_reserve), 'input_token_upper_bound': input_upper,
                       'snapshot_sha256': saved_route['snapshot_sha256'], 'pricing_per_token': saved_route['pricing']}
            attempts.append(attempt)
            op.write_json(path, record)
            start = time.monotonic()
            stage = 'transport'
            try:
                raw = self.client.complete(body)
                usage = raw.get('usage') if isinstance(raw, dict) else None
                cost = usage.get('cost') if isinstance(usage, dict) else None
                safe_raw = foundation._json_safe(raw)
                attempt.update(response=safe_raw, response_id=safe_raw.get('id') if isinstance(safe_raw, dict) else None,
                               returned_model=safe_raw.get('model') if isinstance(safe_raw, dict) else None,
                               routing_metadata=safe_raw.get('openrouter_metadata') if isinstance(safe_raw, dict) else None,
                               reasoning_tokens=op.reasoning_tokens(raw),
                               reported_cost_usd=cost if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 else None)
                op.write_json(path, record)  # Persist billed response before any validation.
                stage = 'route_validation'
                validate_route(attempt)
                stage = 'response_validation'
                text, finish = op.response_text(raw, generation=generation)
                attempt.update(text=text, finish_reason=finish)
                if parse is not None:
                    op.require(finish == 'stop', 'Truncated judge response')
                    stage = 'judgment_parse'
                    attempt['parsed'] = parse(text)
                # Generation text, including empty/capped answers, is retained.
                attempt['status'] = 'ok'
            except Exception as exc:
                diagnostic = diagnostic_context(exc, attempt, stage, attempt.get('failure_diagnostic'))
                attempt.update(status='failed', error='Response or transport failure; details suppressed',
                               failure_diagnostic=diagnostic,
                               setup_error=diagnostic['http_status'] in (400, 401, 402, 403, 404, 405, 422))
            if attempt['reported_cost_usd'] is None and not attempt.get('failure_diagnostic'):
                attempt['failure_diagnostic'] = diagnostic_context(None, attempt, 'billing')
            attempt.update(finished_utc=op.utcnow(), latency_seconds=time.monotonic() - start)
            op.write_json(path, record)
            if attempt.get('failure_diagnostic'):
                op.write_json(self.output / 'last_failure.json', {
                    'request_id': call_id, 'attempt': attempt['attempt'],
                    'failure_diagnostic': attempt['failure_diagnostic'],
                    'accounting': self.account()})
            op.require(attempt['reported_cost_usd'] is not None, 'Unknown billing: saved response requires reconciliation')
            op.require(not attempt.get('routing_mismatch'), 'Returned route mismatch; scheduling stopped')
            op.require(attempt['reported_cost_usd'] <= reserve, 'Request exceeded its reserve; scheduling stopped')
            op.require(not attempt.get('setup_error'), 'API setup error; scheduling stopped')
            if attempt['status'] == 'ok':
                return attempt
        return attempts[-1]

    def reconcile(self, call_id, attempt, cost, note, resolve_setup=False):
        op.require(op.identifier(call_id) and type(attempt) is int and attempt >= 0, 'Invalid attempt')
        op.require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0, 'Invalid reconciled cost')
        op.require(isinstance(note, str) and note.strip(), 'Record the billing evidence in a note')
        path = self.requests / (call_id + '.json')
        record = json.loads(path.read_text())
        item = record['attempts'][attempt]
        item.setdefault('billing_history', []).append({'previous_cost': item.get('reported_cost_usd'),
                                                       'cost': cost, 'note': note, 'utc': op.utcnow()})
        item['reported_cost_usd'] = cost
        if resolve_setup:
            op.require(item.get('setup_error'), 'This attempt has no setup error to resolve')
            item['setup_resolution'] = {'utc': op.utcnow(), 'note': note}
        if item['status'] == 'started' and 'response' not in item:
            item.update(status='failed', error='Interrupted attempt; reconciled without a usable response')
        op.write_json(path, record)
        self._records[call_id] = record
