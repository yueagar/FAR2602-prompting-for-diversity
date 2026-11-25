"""Explicit, separately approved exploratory screening and main execution.

Failed development checks remain failures. This path binds their evidence and
an accepted exception without weakening the original qualification/main gates.
Plans are offline; both paid stages need their own approved record and limits.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import fresh_headroom as fresh
import api_runner as op
import api_data as data
import api_evaluation as ev
import api_runtime as runtime
import api_screen as screening
import study_runner as rp
from api_env import api_key
from research_client import Client

SCHEMA = 'estr-exploratory-adoption-v1'
FREEZE_SCHEMA = 'estr-exploratory-freeze-v1'
QUALIFICATION_SHA256 = fresh.QUALIFICATION_SHA256
QUALIFICATION_MANIFEST_SHA256 = fresh.QUALIFICATION_MANIFEST_SHA256
ADJUDICATION_SHA256 = fresh.ADJUDICATION_SHA256
HEADROOM_SHA256 = '3553472b0c55055b88eb358f79546d46171b5e4d0ad254a4aad395e4ce3c04bd'
HEADROOM_MANIFEST_SHA256 = 'deabdb9b822650955ba1bbb2252485ceffeeafdfa4963484fc9e93f76632aa61'
FRESH_CHECK_DECISION_SHA256 = '5d259bfb0a77cdc675fc86f48d88df3cfa17766ea97a804a3663b4d93d9c3c3a'
PROFILE_SHA256 = 'e633a937f003a4a3fabc1006fd7d61a5679ce6d061f0d58b405815390cb2d3c5'
LIMITS = {'manual_screen': {'max_requests': 60, 'max_cost_usd': 1, 'timeout_seconds': 600},
          'main': {'max_requests': 3888, 'max_cost_usd': 5, 'timeout_seconds': 600}}
EXCEPTIONS = {'fresh_check_failed': True, 'headroom_failed': True, 'single_rater_reference': True}
FLAGS = {'exploratory': True, 'confirmatory': False, 'main_eligible': False, 'development_only': True}
ARM_IDS = ['default', 'naive_creative', 'manual_causal_shift_v1', 'manual_alternative_selection_v1']
SUBSET = {'id': 'development_headroom_le_7', 'tasks': ['curated-0', 'curated-16', 'narrative-dev-01'],
          'selection_basis': 'study default distinct count <= 7', 'measures': ['distinct_count'],
          'pooled_interval': False}
REFS = ('qualification', 'qualification_manifest', 'adjudication', 'headroom',
        'headroom_manifest', 'fresh_check_decision', 'screen_spec')


def config_fingerprint(cfg):
    """Exclude only the commit back-reference to avoid a freeze/hash cycle."""
    return op.fingerprint(cfg | {'freeze_commit': None})


def expected_scope(cfg):
    tasks = data.development_tasks()
    return {'n': 10, 'generator': 'anchor', 'task_ids': sorted(t['id'] for t in tasks),
            'arm_ids': list(ARM_IDS), 'repetitions': 2, 'grouping': 'gemini', 'quality': 'gemini',
            'check': 'gpt', 'check_qualified': False,
            'check_tasks': [t['id'] for t in sorted(tasks,
                key=lambda t: op.fingerprint([cfg['seed'], 'anchor', t['id']]))[:3]]}


def analysis_reporting(include_subset=False):
    return {'primary_task_ids': sorted(t['id'] for t in data.development_tasks()),
            'secondary_subsets': [copy.deepcopy(SUBSET)] if include_subset else []}


def check_reporting(record):
    op.require(op.fingerprint(record) in (op.fingerprint(analysis_reporting(False)), op.fingerprint(analysis_reporting(True))),
               'Declare all ten primary tasks and only the optional per-task headroom subset')


def reference(ref, base, label):
    return fresh.reference(ref, base, label)


def selection(cfg):
    return {'status': 'exploratory_only', 'grouping': 'gemini', 'quality': 'gemini',
            'check': 'gpt', 'check_qualified': False, 'preference': cfg['judge_preference']}


def assert_scope(cfg, authorization, stage):
    """Runner-side defense: this envelope cannot enable an ordinary main run."""
    op.require(stage in LIMITS and isinstance(authorization, dict), 'Exploratory stage authorization required')
    adoption = authorization['adoption']
    op.require(adoption['schema'] == SCHEMA and adoption['stage'] == stage,
               'Exploratory authorization is for a different stage')
    op.require(adoption['approved_by'] == ['study_owner'] and cfg['approved_by'] == ['study_owner'],
               'the study owner must approve this exact exploratory stage')
    op.require(adoption['config_sha256'] == config_fingerprint(cfg), 'Exploratory config changed')
    op.require(adoption['limits'] == cfg['limits'] == LIMITS[stage], 'Exploratory stage limits differ')
    op.require(adoption['scope'] == expected_scope(cfg) and len(cfg['generators']) == 1
               and cfg['generators'][0]['label'] == 'anchor'
               and [a['id'] for a in cfg['arms']] == ARM_IDS and cfg['repetitions'] == 2,
               'Exploratory scope differs from the adopted four-arm n=10 design')
    op.require(adoption['exceptions'] == EXCEPTIONS, 'Retain both failed checks and the single-rater amendment')
    op.require(all(authorization.get(k) is value for k, value in FLAGS.items()), 'Exploratory execution flags differ')
    op.require(authorization['evaluation_signature'] == rp.signature(cfg)
               and authorization['generation_signature'] == rp.generation_signature(cfg), 'Exploratory signatures differ')
    op.require(adoption['code_sha256'] == runtime.code_hashes(), 'Exploratory implementation changed')
    check_reporting(adoption['analysis_reporting'])
    if stage == 'main':
        op.require(adoption['screening'] is not None and adoption['screening_manifest'] is not None,
                   'Exploratory main requires a bound passing screen')


def _check_screen_result(refs, cfg, chosen, spec, main_adoption):
    """Require a completed screen from the same implementation and adopted setup."""
    result = json.loads(refs['screening'].read_text())
    manifest = json.loads(refs['screening_manifest'].read_text())
    source = manifest['specification']
    op.require(refs['screening_manifest'] == refs['screening'].with_name('manifest.json'),
               'Screen manifest must belong to the screening result')
    op.require(source['phase'] == result['phase'] == 'manual_screen' and source['spec'] == result['spec'] == spec,
               'Screen specification differs')
    authorization = source['exploratory_authorization']
    op.require(authorization['adoption']['stage'] == 'manual_screen'
               and authorization['adoption']['approved_by'] == ['study_owner'], 'Screen was not separately authorized')
    op.require(result['exploratory_authorization_sha256'] == op.fingerprint(authorization)
               and all(result.get(k) is value and authorization.get(k) is value for k, value in FLAGS.items()),
               'Screen exploratory provenance differs')
    op.require(manifest['code_sha256'] == result['code_sha256'] == runtime.code_hashes(),
               'Screen implementation differs from the main freeze')
    op.require(source['signature'] == result['signature'] == screening.signature(cfg, chosen, spec),
               'Screen generator, instructions or judge differs')
    screening.check_screen(refs['screening'], cfg, chosen, allow_exploratory=True)
    expected = {(task, arm) for task in spec['task_ids'] for arm in spec['arm_ids']}
    op.require({(b['task'], b['arm']) for b in result['blocks']} == expected
               and len(result['blocks']) == len(expected), 'Incomplete or duplicate screen blocks')
    for block in result['blocks']:
        op.require(block['attempted'] == len(block['outcomes']) == spec['samples_per_task']
                   and all(type(row['pass']) is bool for row in block['outcomes'])
                   and all(row['pass'] == _screen_row_pass(row) for row in block['outcomes'])
                   and block['passes'] == sum(row['pass'] for row in block['outcomes'])
                   and block['pass'] == (block['passes'] >= spec['minimum_passes_per_task'])
                   and block['passes'] >= spec['minimum_passes_per_task'], 'Screen passing counts differ')
    # The adopted evidence must be the same for the screen and main.
    screen_adoption = authorization['adoption']
    op.require(all(screen_adoption[k]['sha256'] == main_adoption[k]['sha256']
                   for k in ('qualification', 'qualification_manifest', 'adjudication', 'headroom',
                             'headroom_manifest', 'fresh_check_decision', 'screen_spec'))
               and all(screen_adoption[k] == main_adoption[k] for k in ('evaluation_profile_sha256', 'development_tasks_sha256',
                                                                       'exceptions', 'scope', 'code_sha256')),
               'Screen and main adoptions bind different evidence or implementation')


def _screen_row_pass(row):
    """Recompute a stored screen row's pass from its saved observations."""
    generation, adherence, diagnostics = row.get('generation') or {}, row.get('adherence'), row.get('diagnostics')
    if generation.get('status') != 'ok' or adherence is None or diagnostics is None:
        return False
    return bool((adherence.get('parsed') or {}).get('pass') and diagnostics['constraint_satisfied'] is not False
                and not diagnostics['flags']['empty'] and not diagnostics['flags']['refusal'])


def validate(config_path, adoption_path, *, stage=None, execute=False, require_screen=False, output=None):
    config_path, adoption_path = Path(config_path).resolve(), Path(adoption_path).resolve()
    cfg, base = rp.load_config(config_path)
    adoption = json.loads(adoption_path.read_text())
    op.keys(adoption, {'schema', 'stage', 'approved_by', 'decision_reference', 'config_sha256', *REFS,
        'screening', 'screening_manifest', 'evaluation_profile_sha256', 'development_tasks_sha256',
        'code_sha256', 'exceptions', 'scope', 'analysis_reporting', 'limits'}, 'exploratory adoption')
    op.require(adoption['schema'] == SCHEMA and adoption['stage'] in LIMITS, 'Wrong exploratory adoption schema/stage')
    stage = stage or adoption['stage']
    op.require(stage == adoption['stage'], 'Requested stage differs from adoption')
    op.require(adoption['approved_by'] in ([], ['study_owner']), 'Adoption approval must be empty or the study owner')
    op.require(isinstance(adoption['decision_reference'], str) and adoption['decision_reference'].strip(),
               'Record the exploratory decision reference')
    op.require(adoption['config_sha256'] == config_fingerprint(cfg), 'Normalized config hash mismatch')
    op.require(adoption['code_sha256'] == runtime.code_hashes(), 'Implementation differs from adoption')
    op.require(adoption['limits'] == cfg['limits'] == LIMITS[stage], 'Stage limits differ')
    op.require(adoption['exceptions'] == EXCEPTIONS, 'Accept and retain both failed checks and single-rater reference')
    op.require(adoption['scope'] == expected_scope(cfg) and len(cfg['generators']) == 1
               and cfg['generators'][0]['label'] == 'anchor'
               and [a['id'] for a in cfg['arms']] == ARM_IDS, 'Exploratory scope differs')
    check_reporting(adoption['analysis_reporting'])
    refs = {name: reference(adoption[name], adoption_path.parent, name) for name in REFS}
    expected_hashes = {'qualification': QUALIFICATION_SHA256, 'qualification_manifest': QUALIFICATION_MANIFEST_SHA256,
                       'adjudication': ADJUDICATION_SHA256, 'headroom': HEADROOM_SHA256,
                       'headroom_manifest': HEADROOM_MANIFEST_SHA256,
                       'fresh_check_decision': FRESH_CHECK_DECISION_SHA256}
    op.require(all(adoption[k]['sha256'] == value for k, value in expected_hashes.items()),
               'Use the preserved qualification, partition, failed headroom and fresh-check evidence')
    op.require(refs['qualification_manifest'] == refs['qualification'].with_name('manifest.json')
               and refs['headroom_manifest'] == refs['headroom'].with_name('manifest.json'),
               'Source manifests must belong to their results')
    qualification = json.loads(refs['qualification'].read_text())
    qmanifest = json.loads(refs['qualification_manifest'].read_text())
    qsource = qmanifest['specification']
    head = json.loads(refs['headroom'].read_text())
    hmanifest = json.loads(refs['headroom_manifest'].read_text())
    hsource = hmanifest['specification']
    op.require(qualification['development_only'] is True and qsource['development_only'] is True
               and qualification['selection']['status'] == 'development_only'
               and qualification['phase'] == qsource['phase'] == 'qualification_10', 'Preserve v2 development-only status')
    op.require(qualification['adjudication_sha256'] == qsource['adjudication_sha256'] == ADJUDICATION_SHA256
               and qsource['adjudication'] == json.loads(refs['adjudication'].read_text()), 'Partition binding differs')
    op.require(qualification['numerical_selection']['grouping'] == qualification['numerical_selection']['quality'] == 'gemini'
               and qualification['summaries']['gemini']['grouping_pass'] is True
               and qualification['summaries']['gemini']['quality']['pass'] is True, 'Hydrogen must pass the preserved v2 development gates')
    op.require(cfg['judges'] == qsource['config']['judges'] == hsource['config']['judges']
               and cfg['generators'] == qsource['config']['generators'] == hsource['config']['generators'],
               'Qualified model routes or judge settings changed')
    op.require(cfg['arms'] == hsource['config']['arms'], 'Keep the existing four instruction strings')
    op.require(cfg.get('evaluation_profile') == qsource['config'].get('evaluation_profile') == ev.TASK_SENSITIVE_PROFILE,
               'Exploratory execution requires the frozen v2 profile')
    op.require(qualification['evaluation_signature'] == head['evaluation_signature'] == rp.signature(cfg),
               'Evaluator signature differs from the source evidence')
    op.require(head['generation_signature'] == rp.generation_signature(cfg)
               == rp.generation_signature(hsource['config']), 'Generation settings differ from fresh headroom')
    profile = ev.profile(ev.TASK_SENSITIVE_PROFILE)
    op.require(qsource['evaluation'] == profile and adoption['evaluation_profile_sha256']
               == op.fingerprint(profile) == PROFILE_SHA256, 'Frozen evaluator prompt text differs')
    op.require(qualification['gates'] == cfg['grouping_gates']['ten'] == qsource['config']['grouping_gates']['ten'],
               'Historical qualification gates changed')
    routes = {j['label']: j for j in cfg['judges']}
    op.require(routes['gemini']['model'] == 'google/gemini-4-hydrogen'
               and routes['gemini']['provider'] == 'google-ai-studio/flex'
               and routes['gpt']['family'] == 'openai', 'Retain Hydrogen primary and Luna supplementary routes')
    op.require(head['phase'] == hsource['phase'] == 'headroom_10' and head['n'] == hsource['n'] == 10
               and head['main_eligible'] is False and head['development_only'] is True
               and head['headroom']['pass'] is False and head['headroom']['next'] == 'stage_20'
               and all(m['complete'] for m in head['headroom']['models'].values()), 'Retain the complete failed headroom result')
    op.require(head['code_sha256'] == hmanifest['code_sha256'], 'Historical headroom implementation binding differs')
    tasks = data.development_tasks()
    op.require(len(tasks) == 10 and adoption['development_tasks_sha256'] == op.fingerprint(tasks)
               and hsource['tasks'] == tasks, 'Permitted development task binding differs')
    op.require(head['check_tasks'] == hsource['check_tasks'] == {'anchor': expected_scope(cfg)['check_tasks']},
               'Prespecified Luna task blocks differ')
    snapshots = rp.route_snapshots(cfg, base, True)
    op.require(all(entry['sha256'] == hsource['snapshots'][key]['sha256'] for key, entry in snapshots.items()),
               'Saved route snapshot changed since fresh headroom')
    spec = json.loads(refs['screen_spec'].read_text())
    screening.validate_spec(spec, cfg, require_approved=execute or require_screen)
    op.require(spec['task_ids'] == ['curated-0', 'narrative-dev-02']
               and spec['arm_ids'] == ARM_IDS[2:] and spec['samples_per_task'] == 5
               and spec['minimum_passes_per_task'] == 4, 'Use the separately proposed two-task 4/5 screen scope')
    chosen = selection(cfg)
    screen_complete = False
    op.require((adoption['screening'] is None) == (adoption['screening_manifest'] is None), 'Bind both screen result and manifest')
    if stage == 'manual_screen':
        op.require(adoption['screening'] is None, 'Screen approval cannot authorize main execution')
    elif adoption['screening'] is not None:
        refs.update({name: reference(adoption[name], adoption_path.parent, name)
                     for name in ('screening', 'screening_manifest')})
        _check_screen_result(refs, cfg, chosen, spec, adoption)
        screen_complete = True
    if stage == 'main' and (execute or require_screen):
        op.require(screen_complete, 'Complete the separately approved screen before freezing or running main')
        from api_report import check_analysis, check_quality_guard
        check_analysis(cfg, require_frozen=True)
        check_quality_guard(cfg, require_support=True, tasks=expected_scope(cfg)['task_ids'])
    inputs = {str(config_path): data.sha(config_path), str(adoption_path): data.sha(adoption_path)}
    inputs.update({str(path): data.sha(path) for path in refs.values()})
    inputs.update({entry['path']: entry['sha256'] for entry in snapshots.values()})
    if output is not None:
        destination = Path(output).resolve()
        protected = [refs['qualification'].parent, refs['headroom'].parent]
        if 'screening' in refs: protected.append(refs['screening'].parent)
        op.require(all(not destination.is_relative_to(path) for path in protected)
                   and all(destination != Path(path) and destination not in Path(path).parents for path in inputs),
                   'Output must preserve source evidence and approved inputs')
    authorization = {'adoption': adoption, 'adoption_sha256': data.sha(adoption_path), 'inputs_sha256': inputs,
                     'evaluation_signature': rp.signature(cfg), 'generation_signature': rp.generation_signature(cfg), **FLAGS}
    if execute:
        assert_scope(cfg, authorization, stage)
        rp.ready(cfg)
    elif require_screen:
        # Scientific settings are approved before freeze; launch approval is
        # added only after reviewing the committed, unchanged specification.
        op.require(cfg['approved_by'] == ['study_owner'], 'Approve scientific settings before the offline freeze')
        rp.ready(cfg)
    return cfg, base, chosen, authorization


def freeze_payload(cfg, base, authorization):
    """The committed payload uses semantic config identity, not its commit pointer."""
    adoption = authorization['adoption']
    op.require(adoption['stage'] == 'main' and adoption['screening'] is not None, 'Main freeze requires a passing screen')
    specification = adoption | {'approved_by': []}
    return {'schema': FREEZE_SCHEMA, 'config': cfg | {'freeze_commit': None}, 'n': 10,
            'adoption_specification': specification,
            'adoption_specification_sha256': op.fingerprint(specification),
            'approval_note': 'Only launch approval is excluded here; execution binds the exact approved adoption bytes.',
            'snapshots': rp.route_snapshots(cfg, base, True), 'code_sha256': runtime.code_hashes(), **FLAGS}


def plan(cfg, authorization):
    stage = authorization['adoption']['stage']
    screen = stage == 'manual_screen'
    return {'stage': stage, 'scope': authorization['adoption']['scope'],
            'generation_calls': 20 if screen else 800, 'primary_judge_calls': 20 if screen else 880,
            'supplementary_calls': 0 if screen else 264, 'initial_calls': 40 if screen else 1944,
            'maximum_attempts': 60 if screen else 3888, 'limits': cfg['limits'],
            'approved': authorization['adoption']['approved_by'] == cfg['approved_by'] == ['study_owner'],
            'screening_bound': authorization['adoption']['screening'] is not None,
            'analysis_reporting': authorization['adoption']['analysis_reporting'], 'minimum_request_start_interval_seconds': 2, **FLAGS,
            'budget_note': 'Hard stage stop, not a completion guarantee; screen and main need separate approvals.'}


RECHECK_EVERY = 20


class DeferredClient(fresh.DeferredClient):
    """Bind inputs before credentials, then recheck code and input hashes.

    Rechecks run before the client is built, before the next request after
    each RECHECK_EVERY calls, and after the stage, including after failure.
    """
    def __init__(self, cfg, authorization):
        super().__init__(cfg, authorization)
        self.stage = authorization['adoption']['stage']
        self.last_start = None
        self.attempts = 0

    def verify_bindings(self):
        op.require(runtime.code_hashes() == self.code, 'Approved implementation changed during the stage')
        op.require(all(data.sha(path) == digest for path, digest in self.inputs.items()), 'An approved input changed during the stage')

    def prepare(self):
        if self.client is None:
            self.verify_bindings()
            self.client = Client(api_key(), self.timeout)

    def before_attempt(self):
        """Called by the journal before it records an attempt, so drift stops cleanly."""
        if self.attempts and self.attempts % RECHECK_EVERY == 0:
            self.verify_bindings()

    def complete(self, body):
        op.require(self.client is not None, 'Client must be prepared after input binding')
        if self.last_start is not None:
            remaining = 2.0 - (time.monotonic() - self.last_start)
            if remaining > 0:
                time.sleep(remaining)
        self.last_start = time.monotonic()
        self.attempts += 1
        progress = self.attempts == 1 or self.attempts % 20 == 0
        if progress:
            print(json.dumps({'event': 'request_started', 'stage': self.stage, 'attempt': self.attempts}), flush=True)
        response = self.client.complete(body)
        if progress:
            print(json.dumps({'event': 'request_returned', 'stage': self.stage, 'attempt': self.attempts}), flush=True)
        return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'screen', 'freeze', 'run'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--adoption', required=True)
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        stage = {'screen': 'manual_screen', 'freeze': 'main', 'run': 'main'}.get(args.command)
        execute = args.command in ('screen', 'run')
        op.require(not execute or (args.execute and args.output), 'Paid stages require --execute and --output')
        cfg, base, chosen, authorization = validate(args.config, args.adoption, stage=stage, execute=execute,
            require_screen=args.command == 'freeze', output=args.output)
        if args.command == 'plan':
            print(json.dumps(plan(cfg, authorization), indent=2)); return 0
        if args.command == 'freeze':
            destination = (base / cfg['freeze_file']).resolve()
            op.require(args.output and Path(args.output).resolve() == destination, '--output must match config.freeze_file')
            op.require(not destination.exists(), 'Preserve the existing freeze; use a new version')
            op.write_json(destination, freeze_payload(cfg, base, authorization)); return 0
        client = DeferredClient(cfg, authorization)
        try:
            if args.command == 'screen':
                spec_path = (Path(args.adoption).resolve().parent / authorization['adoption']['screen_spec']['path']).resolve()
                result = screening.screen(cfg, base, chosen, spec_path, args.output, client,
                                          exploratory_authorization=authorization)
            else:
                # run_comparison validates the committed freeze itself.
                result = rp.run_comparison(cfg, base, chosen, args.output, client, n=10, phase='main',
                                           exploratory_authorization=authorization)
        except BaseException:
            # Report input or code drift even when the stage stops early.
            if client.client is not None:
                try:
                    client.verify_bindings()
                except ValueError as drift:
                    print(json.dumps({'event': 'binding_check_after_failure', 'error': str(drift)}), flush=True)
            raise
        client.verify_bindings()
        print(json.dumps({'output': str(Path(args.output).resolve()), 'accounting': result['accounting'], **FLAGS}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        print(f'Error: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
