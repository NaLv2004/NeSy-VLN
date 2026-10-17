"""Read already-completed development traces; never run a policy."""
import json
from pathlib import Path
from collections import Counter
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'diagnostics/raw_boundary_v19_navigation_audit.json';assert not out.exists()
reports=[]
for update in [200,400,600]:
    folder=root/'run_raw_boundary_v19'/f'development_step_{update:06d}'
    total=Counter();first=Counter();lengths=Counter();examples=[]
    for path in sorted(folder.glob('*/trace.json')):
        r=json.loads(path.read_text());ev=json.loads((path.parent/'evaluation.json').read_text());steps=r['steps'];programs=r['programs']
        actions=[x['action'] for x in steps];total['episodes']+=1;total['successes']+=ev['success']
        lengths[len(actions)]+=1
        if actions:
            first[actions[0]]+=1
            total['stopped']+=actions[-1]==0
            total['stopped_unsuccessfully']+=(actions[-1]==0 and not ev['success'])
            total['stopped_on_first_action']+=actions==[0]
            total['consecutive_repeat_actions']+=sum(x==y for x,y in zip(actions,actions[1:]))
            total['adjacent_action_pairs']+=max(len(actions)-1,0)
        total['planning_failures']+=ev['planning_failed']
        total['no_action']+=not actions
        for program in programs:
            plan=program['plan'];total['programs']+=1
            if plan['found']:
                total['exact_plans']+=1
                total['plan_returns_to_initial_state']+=plan['states'][0]==plan['states'][-1]
        if len(examples)<5 and actions==[0]:
            examples.append(dict(episode=ev['episode_id'],metrics=r['metrics'],first_program=programs[0]))
    assert total['episodes']==80
    reports.append(dict(update=update,counts=dict(total),first_actions=dict(first),length_histogram=dict(lengths),immediate_stop_examples=examples))
out.write_text(json.dumps(dict(no_new_evaluation=True,reports=reports),indent=2))
print(json.dumps([dict(update=r['update'],counts=r['counts'],first_actions=r['first_actions'],length_histogram=r['length_histogram']) for r in reports]))
