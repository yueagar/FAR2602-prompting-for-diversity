"""Separately adopted, bounded headroom for the fresh human grouping check.

Planning is offline. Execution needs explicit approval in both input files and
--execute; it does not turn the historical development qualification into a
main-study selection. Credentials are loaded only after the run is bound.
"""
import argparse
import json
from pathlib import Path

import api_runner as op
import api_data as data
import api_evaluation as ev
import api_runtime as runtime
import study_runner as rp
from api_env import api_key
from research_client import Client

SCHEMA = 'estr-fresh-headroom-adoption-v1'
QUALIFICATION_SHA256 = 'e1ea3afbf91b8f0c5f87cb4c5f34f74ad935333cf8ee76363a169ed37af47349'
QUALIFICATION_MANIFEST_SHA256 = 'b042f5813e70077a5539fa740fb6222ce5eb2b18acca13a1ff0489c00c46c8c0'
ADJUDICATION_SHA256 = '587dba4821691df5290c6bf07437c392bf599c92274c5f954928041964a89000'
HUMAN_TASKS = ['curated-6', 'curated-7', 'curated-8', 'narrative-dev-01', 'narrative-dev-02']
LIMITS = {'max_requests': 220, 'max_cost_usd': 1, 'timeout_seconds': 600}
FRESH_CHECK_RULES = {'complete_sets': 5, 'count_mae_max': 0.5, 'count_worst_max': 2,
    'false_merge_rate_max': 0.05, 'false_split_rate_max': 0.05,
    'zero_pair_denominator': 'inconclusive', 'order_repeat_test': False}


def reference(ref, base, label):
    op.keys(ref, {'path', 'sha256'}, label)
    op.require(isinstance(ref['path'], str) and ref['path'].strip(), f'{label} path required')
    path = (base / ref['path']).resolve()
    op.require(path.is_file() and data.sha(path) == ref['sha256'], f'{label} hash mismatch')
    return path


def validate(config_path, adoption_path, *, execute=False, output=None):
    """Validate local evidence and exact scope without accessing credentials."""
    config_path, adoption_path = Path(config_path).resolve(), Path(adoption_path).resolve()
    cfg, base = rp.load_config(config_path)
    adoption = json.loads(adoption_path.read_text())
    op.keys(adoption, {'schema', 'approved_by', 'decision_reference', 'agreement',
        'config_sha256', 'qualification', 'qualification_manifest', 'adjudication',
        'evaluation_profile_sha256', 'development_tasks_sha256', 'code_sha256',
        'fresh_check_rules', 'stage', 'limits'}, 'fresh-check adoption')
    op.require(adoption['schema'] == SCHEMA, 'Wrong fresh-check adoption schema')
    op.require(adoption['approved_by'] in ([], ['study_owner']), 'Adoption approval must be empty or the study owner')
    op.require(isinstance(adoption['decision_reference'], str) and adoption['decision_reference'].strip(),
               'Record the fresh-check decision reference')
    op.require(adoption['config_sha256'] == data.sha(config_path), 'Config hash mismatch')
    op.require(adoption['code_sha256'] == runtime.code_hashes(), 'Implementation differs from the proposed/approved adoption')
    op.require(adoption['fresh_check_rules'] == FRESH_CHECK_RULES, 'Fresh-check operating rules differ')
    op.require(cfg.get('evaluation_profile') == ev.TASK_SENSITIVE_PROFILE, 'Fresh check requires the v2 profile')
    op.require(adoption['limits'] == cfg['limits'] == LIMITS, 'Fresh-check limits must be 220 attempts, US$1, 600 seconds')
    op.require(len(cfg['generators']) == 1, 'Fresh check permits exactly one generator')
    stage = {'phase': 'headroom_10', 'n': 10, 'generator': cfg['generators'][0]['label'],
        'arm': 'default', 'repetitions': 1, 'grouping': 'gemini', 'quality': 'gemini',
        'check': None, 'human_check_tasks': HUMAN_TASKS}
    op.require(adoption['stage'] == stage, 'Fresh-check scope must match default-only headroom_10 and the fixed five tasks')
    op.require(next(a for a in cfg['arms'] if a['id'] == 'default') ==
               {'id': 'default', 'instruction': '', 'decoding_overrides': {}}, 'Default arm changed')

    refs = {name: reference(adoption[name], adoption_path.parent, name)
            for name in ('agreement', 'qualification', 'qualification_manifest', 'adjudication')}
    op.require(adoption['qualification']['sha256'] == QUALIFICATION_SHA256, 'Use the frozen v2 qualification')
    op.require(adoption['qualification_manifest']['sha256'] == QUALIFICATION_MANIFEST_SHA256,
               'Use the frozen v2 qualification manifest')
    op.require(adoption['adjudication']['sha256'] == ADJUDICATION_SHA256, 'Use the frozen v2 adjudication')
    op.require(refs['qualification_manifest'] == refs['qualification'].with_name('manifest.json'),
               'Qualification manifest must belong to the source run')
    qualification = json.loads(refs['qualification'].read_text())
    manifest = json.loads(refs['qualification_manifest'].read_text())
    source = manifest['specification']
    op.require(manifest['schema'] == 'openrouter-research-run-v1'
               and source['phase'] == qualification['phase'] == 'qualification_10', 'Wrong qualification evidence')
    op.require(qualification['development_only'] is True and source['development_only'] is True
               and qualification['selection']['status'] == 'development_only', 'Preserve the development-only qualification')
    # Qualification results predate result-level code hashes. The separately
    # hash-pinned source manifest carries the historical implementation binding.
    op.require(isinstance(manifest['code_sha256'], dict) and bool(manifest['code_sha256']),
               'Qualification manifest has no implementation binding')
    op.require(source['adjudication_sha256'] == qualification['adjudication_sha256'] == ADJUDICATION_SHA256
               and source['adjudication'] == json.loads(refs['adjudication'].read_text()), 'Adjudication binding differs')
    op.require(cfg['judges'] == source['config']['judges'], 'Judge routes or parameters changed since qualification')
    op.require(cfg['generators'] == source['config']['generators'], 'Use the qualified generator route')
    op.require(cfg['evaluation_profile'] == source['config'].get('evaluation_profile'), 'Qualification profile differs')
    op.require(qualification['evaluation_signature'] == rp.signature(cfg) == rp.signature(source['config']),
               'Qualification evaluator signature differs')
    op.require(qualification['gates'] == cfg['grouping_gates']['ten'] == source['config']['grouping_gates']['ten'],
               'Qualification gates changed')
    evaluation = ev.profile(ev.TASK_SENSITIVE_PROFILE)
    op.require(source['evaluation'] == evaluation
               and adoption['evaluation_profile_sha256'] == op.fingerprint(evaluation), 'Actual evaluator prompt text differs')
    primary = next((j for j in cfg['judges'] if j['label'] == 'gemini'), None)
    op.require(primary is not None and primary['model'] == 'google/gemini-4-hydrogen'
               and primary['provider'] == 'google-ai-studio/flex', 'Adopted primary must be Hydrogen on the frozen provider')
    chosen = qualification['numerical_selection']
    op.require(chosen['status'] == 'selected' and chosen['grouping'] == chosen['quality'] == 'gemini'
               and qualification['summaries']['gemini']['grouping_pass'] is True
               and qualification['summaries']['gemini']['quality']['pass'] is True, 'Hydrogen did not pass both development gates')
    tasks = data.development_tasks()
    op.require(len(tasks) == 10 and set(HUMAN_TASKS) <= {task['id'] for task in tasks}
               and adoption['development_tasks_sha256'] == op.fingerprint(tasks), 'Development task binding differs')
    snapshots = rp.route_snapshots(cfg, base, True)
    for judge in cfg['judges']:
        key = 'j-' + judge['label']
        op.require(snapshots[key]['sha256'] == source['snapshots'][key]['sha256'], 'Qualified judge snapshot changed')
    inputs = {str(config_path): data.sha(config_path), str(adoption_path): data.sha(adoption_path)}
    inputs.update({str(path): data.sha(path) for path in refs.values()})
    inputs.update({entry['path']: entry['sha256'] for entry in snapshots.values()})
    if output is not None:
        destination = Path(output).resolve()
        op.require(all(destination != Path(path) and destination not in Path(path).parents for path in inputs),
                   'Output must not contain or replace frozen evidence or inputs')
        op.require(destination != refs['qualification'].parent, 'Preserve the source qualification run')
    if execute:
        op.require(adoption['approved_by'] == ['study_owner'], 'the study owner must approve this new paid stage and US$1 cap')
        rp.ready(cfg)
    selection = {'status': 'fresh_check_only', 'grouping': 'gemini', 'quality': 'gemini',
                 'check': None, 'check_qualified': False, 'preference': cfg['judge_preference']}
    authorization = {'adoption': adoption, 'adoption_sha256': data.sha(adoption_path), 'inputs_sha256': inputs,
                     'evaluation_signature': rp.signature(cfg), 'generation_signature': rp.generation_signature(cfg),
                     'main_eligible': False}
    return cfg, base, selection, authorization


def plan(cfg, authorization):
    return {'phase': 'headroom_10', 'n': 10, 'arm': 'default', 'repetitions': 1,
        'generator': cfg['generators'][0], 'generation': cfg['generation'], 'seed': cfg['seed'],
        'grouping_judge': next(j for j in cfg['judges'] if j['label'] == 'gemini'),
        'generation_calls': 100, 'grouping_calls': 10, 'quality_calls': 0, 'supplementary_calls': 0,
        'initial_calls': 110, 'maximum_attempts': 220, 'limits': cfg['limits'],
        'human_check_tasks': HUMAN_TASKS, 'human_responses': 50, 'main_eligible': False,
        'fresh_check_rules': FRESH_CHECK_RULES,
        'approved': authorization['adoption']['approved_by'] == ['study_owner'] and 'study_owner' in cfg['approved_by'],
        'authorization_sha256': op.fingerprint(authorization),
        'budget_note': 'Hard stop; the cap does not guarantee completion. Existing retry and unknown-billing stops remain.'}


class DeferredClient:
    """Prepare after manifest binding, before any journaled transmission."""
    def __init__(self, cfg, authorization):
        self.timeout = cfg['limits']['timeout_seconds']
        self.inputs = authorization['inputs_sha256']
        self.code = authorization['adoption']['code_sha256']
        self.client = None

    def prepare(self):
        if self.client is None:
            op.require(runtime.code_hashes() == self.code, 'Approved implementation changed before transmission')
            op.require(all(data.sha(path) == digest for path, digest in self.inputs.items()), 'An approved input changed before transmission')
            self.client = Client(api_key(), self.timeout)

    def complete(self, body):
        op.require(self.client is not None, 'Client must be prepared after input binding')
        return self.client.complete(body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--adoption', required=True)
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        execute = args.command == 'run'
        op.require(not execute or (args.execute and args.output), 'Run requires --execute and a new --output directory')
        cfg, base, selected, authorization = validate(args.config, args.adoption, execute=execute, output=args.output)
        if not execute:
            print(json.dumps(plan(cfg, authorization), indent=2))
            return 0
        result = rp.run_comparison(cfg, base, selected, args.output, DeferredClient(cfg, authorization),
                                   n=10, phase='headroom_10', stage_authorization=authorization)
        print(json.dumps({'output': str(Path(args.output).resolve()), 'accounting': result['accounting'], 'main_eligible': False}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        print(f'Error: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
