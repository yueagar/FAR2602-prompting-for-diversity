"""Frozen factorial route and allowance receipts; imports are entirely offline.

Only explicitly approved ``check_live`` performs five GETs: three public route
listings and the existing key/account allowances. Never persist raw key data.
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

import api_runner as op
import api_data as data
from api_env import api_key
from rejudge_preflight import _account_check, _fresh, _number, _route_check

SCHEMA = 'temperature-instruction-factorial-preflight-v1'
ROLES = ('generator', 'primary', 'check')
FILES = ('account.json',) + tuple(f'endpoint-{role}.json' for role in ROLES)


def verify_receipt(plan_path, receipt_dir, *, expected_freeze_commit=None, now=None):
    """Require fresh, immutable three-route evidence and full stage-cap allowance.

    Allowance covers the cap, not the sum of hypothetical per-request reserves.
    On resumption the journal still retains all historical spending and attempts.
    """
    import factorial_experiment as experiment
    plan_path, root = Path(plan_path).resolve(), Path(receipt_dir).resolve()
    plan = experiment.verify_plan(plan_path)
    receipt_path = root / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    op.require(receipt.get('schema') == SCHEMA and receipt.get('passed') is True
               and receipt.get('plan_sha256') == data.sha(plan_path),
               'Preflight receipt does not bind this approved plan')
    op.require(re.fullmatch(r'[0-9a-f]{40}', receipt.get('freeze_commit', '')) is not None
               and (expected_freeze_commit is None or
                    receipt['freeze_commit'] == expected_freeze_commit),
               'Preflight receipt does not bind the expected freeze commit')
    expected = {str(root / name) for name in FILES}
    op.require(set(receipt['files_sha256']) == expected
               and all(data.sha(p) == digest for p, digest in receipt['files_sha256'].items()),
               'Preflight evidence changed or refers outside this receipt directory')
    now = now or datetime.now(timezone.utc)
    account = json.loads((root / 'account.json').read_text())
    for stamp in (receipt['checked_utc'], account['checked_utc']):
        _fresh(stamp, now)
    _account_check(account, plan['limits']['max_cost_usd'])
    for role in ROLES:
        saved = json.loads(Path(plan['snapshots'][role]['path']).read_text())
        fresh = json.loads((root / f'endpoint-{role}.json').read_text())
        _fresh(fresh['fetched_utc'], now)
        _route_check(saved, fresh)
    return receipt | {'files_sha256': receipt['files_sha256'] |
                      {str(receipt_path): data.sha(receipt_path)}}


def _read_account(key):
    """Keep only numeric billing counters needed for allowance/recovery evidence."""
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
            for name in ('usage', 'usage_daily', 'usage_weekly', 'usage_monthly'):
                if name in value:
                    op.require(_number(value[name]), 'Malformed key usage evidence')
                    facts[name] = value[name]
        else:
            op.require(all(_number(value.get(name)) for name in ('total_credits', 'total_usage')),
                       'Malformed account credit evidence')
            facts['credits_total'] = value['total_credits']
            facts['credits_usage'] = value['total_usage']
            facts['credits_remaining'] = value['total_credits'] - value['total_usage']
    facts['checked_utc'] = op.utcnow()
    return facts


def check_live(plan_path, output, *, approved_plan_sha256, freeze_commit, approve_live=False):
    import factorial_experiment as experiment
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    op.require(approve_live is True and approved_plan_sha256 == data.sha(plan_path),
               'Explicit scoped approval and approved plan hash required for live metadata checks')
    experiment.verify_live_freeze(plan_path, approved_plan_sha256, freeze_commit, approve_live)
    plan = experiment.verify_plan(plan_path)
    source = Path(plan['source_run']).resolve()
    op.require(output != source and source not in output.parents
               and output not in source.parents and not output.exists(),
               'Use a new preflight directory outside preserved source evidence')
    output.mkdir(parents=True)
    try:
        for role in ROLES:
            fresh = data.fetch_snapshot(plan['routes'][role], output / f'endpoint-{role}.json')
            saved = json.loads(Path(plan['snapshots'][role]['path']).read_text())
            _route_check(saved, fresh)
        account = _read_account(api_key())
        _account_check(account, plan['limits']['max_cost_usd'])
        op.write_json(output / 'account.json', account)
    except Exception:
        raise RuntimeError('Live preflight failed; preserve partial evidence; no model call was made') from None
    receipt = {'schema': SCHEMA, 'passed': True, 'checked_utc': op.utcnow(),
               'plan_sha256': data.sha(plan_path), 'freeze_commit': freeze_commit,
               'files_sha256': {str(output / name): data.sha(output / name) for name in FILES}}
    op.write_json(output / 'receipt.json', receipt)
    return verify_receipt(plan_path, output, expected_freeze_commit=freeze_commit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'verify'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--receipt-dir', required=True)
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--freeze-commit')
    parser.add_argument('--approve-live', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'check':
            result = check_live(args.plan, args.receipt_dir,
                                approved_plan_sha256=args.approved_plan_sha256,
                                freeze_commit=args.freeze_commit, approve_live=args.approve_live)
        else:
            result = verify_receipt(args.plan, args.receipt_dir,
                                    expected_freeze_commit=args.freeze_commit)
        print(json.dumps({'passed': True, 'plan_sha256': result['plan_sha256'],
                          'receipt_dir': str(Path(args.receipt_dir).resolve())}))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError):
        print('Preflight stopped; inspect local evidence. No model request was submitted.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
