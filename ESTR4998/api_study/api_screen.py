"""Development-only manual-instruction screening; never ranks diversity.

The template's two tasks, five samples and 4/5 gate remain a proposal until
the study owner approves the exact screen specification. This is a task-adherence check,
not human validation, a diversity result or a main-run quality guard.
"""
import json
from pathlib import Path

import api_runner as op
import api_data as data
import api_evaluation as ev
from api_runtime import Journal, bind_run, run_lock


def validate_spec(spec, cfg, *, require_approved=True):
    op.keys(spec, {'version','approved_by','decision_reference','task_ids','arm_ids','samples_per_task',
                  'minimum_passes_per_task','judge_prompt'}, 'manual screen')
    op.require(spec['approved_by'] in ([], ['study_owner']) and (not require_approved or spec['approved_by'] == ['study_owner']),
               'the study owner approves the exact manual-screen specification')
    op.require(isinstance(spec['version'],str) and spec['version'].strip() and isinstance(spec['decision_reference'],str)
               and spec['decision_reference'].strip(), 'Record screen version and decision')
    op.require(isinstance(spec['task_ids'],list) and len(set(spec['task_ids']))==len(spec['task_ids']) and spec['task_ids'], 'Unique development tasks required')
    tasks={t['id']:t for t in data.development_tasks()}
    op.require(set(spec['task_ids']) <= tasks.keys(), 'Screening accepts only the frozen development tasks')
    arms={a['id']:a for a in cfg['arms']}
    op.require(isinstance(spec['arm_ids'],list) and len(set(spec['arm_ids']))==len(spec['arm_ids']) and spec['arm_ids'], 'Unique screen arms required')
    op.require(set(spec['arm_ids']) <= arms.keys(), 'Screen arms must come from frozen candidate strings')
    op.require({'manual_causal_shift_v1','manual_alternative_selection_v1'} <= set(spec['arm_ids']), 'Screen both manual candidates')
    op.require(op.integer(spec['samples_per_task']) and type(spec['minimum_passes_per_task']) is int
               and 1 <= spec['minimum_passes_per_task'] <= spec['samples_per_task'], 'Invalid screening gate')
    op.require(isinstance(spec['judge_prompt'],str) and spec['judge_prompt'].strip(), 'Freeze the task-adherence rubric')
    return [tasks[t] for t in spec['task_ids']], [arms[a] for a in spec['arm_ids']]


def signature(cfg, chosen, spec):
    tasks, arms=validate_spec(spec,cfg)
    judge=next(j for j in cfg['judges'] if j['label']==chosen['quality'])
    return op.fingerprint({'spec':spec,'tasks':tasks,'arms':arms,'generator':cfg['generators'][0],
                           'generation':cfg['generation'],'seed':cfg['seed'],'judge':judge})


def parse_adherence(text):
    result=ev.strict_json(text)
    op.keys(result,{'pass','reason'},'task-adherence reply')
    op.require(type(result['pass']) is bool and isinstance(result['reason'],str) and result['reason'].strip(), 'Expected a boolean pass and a nonempty reason')
    return result


def screen(cfg,base,chosen,spec_path,output,client, *, exploratory_authorization=None):
    if exploratory_authorization is not None or chosen.get('status') == 'exploratory_only':
        from exploratory_main import assert_scope, selection as adopted_selection
        op.require(exploratory_authorization is not None, 'Exploratory screening authorization required')
        assert_scope(cfg, exploratory_authorization, 'manual_screen')
        op.require(chosen == adopted_selection(cfg), 'Exploratory screen judge selection differs')
        # Bind the spec file here too, not only in the command-line wrapper.
        adopted_spec = exploratory_authorization['adoption']['screen_spec']
        spec_key = str(Path(spec_path).resolve())
        op.require(exploratory_authorization['inputs_sha256'].get(spec_key) == adopted_spec['sha256'] == data.sha(spec_path),
                   'Screen specification differs from the adopted file')
    spec=json.loads(Path(spec_path).read_text())
    tasks,arms=validate_spec(spec,cfg)
    generator=cfg['generators'][0]
    judge=next(j for j in cfg['judges'] if j['label']==chosen['quality'])
    generation_count=len(tasks)*len(arms)*spec['samples_per_task']
    op.require(2*generation_count <= cfg['limits']['max_requests'], 'Screen exceeds request budget')
    op.require(generation_count*(generator['reserve_usd']+judge['reserve_usd']) <= cfg['limits']['max_cost_usd'], 'Screen exceeds reserved-cost budget')
    snapshots={'generator':data.endpoint_check(generator,cfg['generation'],base,True),
               'judge':data.endpoint_check(judge,judge['parameters'],base,True)}
    stamp=signature(cfg,chosen,spec)
    schema={'type':'object','properties':{'pass':{'type':'boolean'},'reason':{'type':'string'}},
            'required':['pass','reason'],'additionalProperties':False}
    specification={'phase':'manual_screen','signature':stamp,'spec':spec,'tasks':tasks,'arms':arms,'snapshots':snapshots}
    if exploratory_authorization is not None:
        specification.update(config=cfg, selection=chosen, exploratory_authorization=exploratory_authorization)
    with run_lock(output):
        manifest=bind_run(output,specification)
        if exploratory_authorization is not None and hasattr(client, 'prepare'):
            client.prepare()
        journal=Journal(output,client,cfg['limits'],base)
        blocks=[]
        for task in tasks:
            for arm in arms:
                outcomes=[]
                for index in range(spec['samples_per_task']):
                    identity=op.fingerprint([task['id'],arm['id'],index])[:20]
                    parameters=cfg['generation']|arm['decoding_overrides']
                    if 'seed' in parameters:
                        parameters=parameters|{'seed':int(op.fingerprint([cfg['seed'],'manual_screen',task['id'],index])[:8],16)&0x7fffffff}
                    content=task['text']+('\n\nAdditional instruction:\n'+arm['instruction'] if arm['instruction'] else '')
                    generation=journal.call('screen-g-'+identity,generator,op.payload(generator,[{'role':'user','content':content}],parameters),
                                            {'role':'generation','task':task['id'],'arm':arm['id'],'index':index,'phase':'manual_screen'})
                    row={'generation':generation,'pass':False,'adherence':None,'diagnostics':None}
                    if generation['status']=='ok':
                        body=op.payload(judge,[{'role':'system','content':spec['judge_prompt']},
                            {'role':'user','content':ev.messages(task['text'],[generation['text']],'quality')[1]['content']}],judge['parameters'])
                        if judge['json_schema']:
                            body['response_format']={'type':'json_schema','json_schema':{'name':'task_adherence','strict':True,'schema':schema}}
                        else:
                            body['messages'][0]['content']+='\nReturn JSON only: {"pass":true,"reason":"..."}.'
                        judged=journal.call('screen-j-'+identity,judge,body,
                                            {'role':'task_adherence','task':task['id'],'arm':arm['id'],'index':index},parse=parse_adherence)
                        diagnostic=data.text_diagnostics(task['id'],generation['text'],generation['finish_reason'])
                        row.update(adherence=judged,diagnostics=diagnostic)
                        row['pass']=bool(judged.get('parsed',{}).get('pass') and diagnostic['constraint_satisfied'] is not False
                                         and not diagnostic['flags']['empty'] and not diagnostic['flags']['refusal'])
                    outcomes.append(row)
                successes=sum(r['pass'] for r in outcomes)
                blocks.append({'task':task['id'],'arm':arm['id'],'passes':successes,'attempted':len(outcomes),
                               'pass':successes>=spec['minimum_passes_per_task'],'outcomes':outcomes})
        result={'phase':'manual_screen','signature':stamp,'spec':spec,'blocks':blocks,
                'pass':all(b['pass'] for b in blocks),'accounting':journal.account(),
                'note':'Task-adherence screen only; no diversity ranking, no main results, no human validation.'}
        if exploratory_authorization is not None:
            from exploratory_main import FLAGS
            result.update(**FLAGS, code_sha256=manifest['code_sha256'],
                          exploratory_authorization_sha256=op.fingerprint(exploratory_authorization))
        op.write_json(Path(output)/'screen.json',result)
        return result


def check_screen(path,cfg,chosen,*,allow_exploratory=False):
    op.require(path,'Pass the completed --screening result before freezing the manual arms')
    result=json.loads(Path(path).read_text())
    # An exploratory screen must never satisfy the ordinary gated freeze.
    op.require(allow_exploratory or not (result.get('exploratory') is True or result.get('main_eligible') is False
                                         or 'exploratory_authorization_sha256' in result),
               'Exploratory screen results cannot authorize the ordinary main freeze')
    op.require(result['phase']=='manual_screen' and result['signature']==signature(cfg,chosen,result['spec']), 'Manual-screen configuration or instructions changed')
    op.require(result['pass'] and result['blocks'] and all(b['pass'] for b in result['blocks']), 'Both manual instructions must pass on every screening task')
    return data.sha(path)
