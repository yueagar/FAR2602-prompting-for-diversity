"""Bounded route/allowance check for the separately approved rejudging stage.

Import and verify are offline. Only check_live with explicit approval reads the
existing key and makes three GET requests: one public route listing, key facts,
and credits. Raw key responses and credentials are never persisted or printed.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from urllib import request

import api_runner as op
import api_data as data
from api_env import api_key

SCHEMA = 'saved-response-rejudge-preflight-v1'
CRITICAL = ('tag', 'model_id', 'provider_name', 'name', 'service_tier',
            'quantization', 'pricing', 'context_length', 'max_prompt_tokens',
            'max_completion_tokens')


def _fresh(stamp, now):
    value = datetime.fromisoformat(stamp)
    op.require(value.tzinfo is not None, 'Preflight timestamps require a timezone')
    age = (now - value).total_seconds()
    op.require(0 <= age < 3600, 'Refresh preflight evidence before execution')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _account_check(account, cap):
    op.require(all(_number(account.get(k)) for k in
                   ('limit', 'limit_remaining', 'credits_remaining')),
               'Missing or invalid numeric allowance evidence')
    op.require(account['limit'] == 50 and account.get('limit_reset') == 'monthly'
               and account['limit_remaining'] >= cap
               and account['credits_remaining'] >= cap,
               'Monthly key boundary or available allowance cannot cover the stage cap')


def _route_check(saved, fresh):
    op.require(all(saved.get(k) == fresh.get(k) for k in ('model', 'provider')),
               'Fresh route identity differs from the prepared route')
    a, b = saved['endpoint'], fresh['endpoint']
    op.require(all(a.get(k) == b.get(k) for k in CRITICAL)
               and sorted(a['supported_parameters']) == sorted(b['supported_parameters'])
               and saved.get('raw', {}).get('data', {}).get('canonical_slug')
               == fresh.get('raw', {}).get('data', {}).get('canonical_slug'),
               'Fresh route settings, pricing or limits changed; review and prepare a new version')
    op.require(b.get('status') == 0, 'Fresh route is not reported available')


def verify_receipt(plan_path, receipt_dir, *, now=None):
    """Check the immutable evidence and full-cap allowance before a live launch.

    Full-cap coverage is conservative on resume: earlier spend is retained in
    the stage journal, never subtracted from account usage or reset by this check.
    Freshness applies at launch; a long-running stage preserves hashes thereafter.
    """
    import rejudge_saved
    plan_path, root = Path(plan_path).resolve(), Path(receipt_dir).resolve()
    plan = rejudge_saved.verify_plan(plan_path)
    receipt_path = root / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    op.require(receipt.get('schema') == SCHEMA and receipt.get('passed') is True
               and receipt.get('plan_sha256') == data.sha(plan_path),
               'Preflight receipt does not bind this approved plan')
    expected = {str(root / name) for name in ('account.json', 'endpoint.json')}
    op.require(set(receipt['files_sha256']) == expected
               and all(data.sha(p) == digest for p, digest in receipt['files_sha256'].items()),
               'Preflight evidence changed or refers outside this receipt directory')
    now = now or datetime.now(timezone.utc)
    account = json.loads((root / 'account.json').read_text())
    fresh = json.loads((root / 'endpoint.json').read_text())
    for stamp in (receipt['checked_utc'], account['checked_utc'], fresh['fetched_utc']):
        _fresh(stamp, now)
    _account_check(account, plan['limits']['max_cost_usd'])
    saved = json.loads(Path(plan['snapshots']['primary']['path']).read_text())
    _route_check(saved, fresh)
    return receipt | {'files_sha256': receipt['files_sha256'] |
                     {str(receipt_path): data.sha(receipt_path)}}


def _read_account(key):
    opener = request.build_opener(op.NoRedirect())
    facts = {}
    for endpoint in ('key', 'credits'):
        req = request.Request('https://openrouter.ai/api/v1/' + endpoint,
                              headers={'Authorization': 'Bearer ' + key})
        with opener.open(req, timeout=30) as response:
            value = json.loads(response.read())['data']
        if endpoint == 'key':
            facts.update({name: value.get(name) for name in
                          ('limit', 'limit_remaining', 'limit_reset')})
        else:
            op.require(all(_number(value.get(name)) for name in ('total_credits', 'total_usage')),
                       'Malformed account credit evidence')
            facts['credits_remaining'] = value['total_credits'] - value['total_usage']
    facts['checked_utc'] = op.utcnow()
    return facts


def check_live(plan_path, output, *, approved_plan_sha256, approve_live=False):
    import rejudge_saved
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    op.require(approve_live and approved_plan_sha256 == data.sha(plan_path),
               'Explicit scoped approval and approved plan hash required for live metadata checks')
    plan = rejudge_saved.verify_plan(plan_path)
    source = Path(plan['source_run']).resolve()
    op.require(output != source and source not in output.parents
               and output not in source.parents and not output.exists(),
               'Use a new preflight directory outside preserved source evidence')
    output.mkdir(parents=True)
    try:
        fresh = data.fetch_snapshot(plan['route'], output / 'endpoint.json')
        saved = json.loads(Path(plan['snapshots']['primary']['path']).read_text())
        _route_check(saved, fresh)
        account = _read_account(api_key())
        _account_check(account, plan['limits']['max_cost_usd'])
        op.write_json(output / 'account.json', account)
    except Exception:
        raise RuntimeError('Live preflight failed; preserve partial evidence; no model call was made') from None
    receipt = {'schema': SCHEMA, 'passed': True, 'checked_utc': op.utcnow(),
               'plan_sha256': data.sha(plan_path),
               'files_sha256': {str(output / name): data.sha(output / name)
                                for name in ('account.json', 'endpoint.json')}}
    op.write_json(output / 'receipt.json', receipt)
    return verify_receipt(plan_path, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'verify'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--receipt-dir', required=True)
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--approve-live', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'check':
            result = check_live(args.plan, args.receipt_dir,
                                approved_plan_sha256=args.approved_plan_sha256,
                                approve_live=args.approve_live)
        else:
            result = verify_receipt(args.plan, args.receipt_dir)
        print(json.dumps({'passed': True, 'plan_sha256': result['plan_sha256'],
                          'receipt_dir': str(Path(args.receipt_dir).resolve())}))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError):
        print('Preflight stopped; inspect local evidence. No model request was submitted.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
