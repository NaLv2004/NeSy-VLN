"""Audit saved navigation traces, without rerunning navigation or using labels for training."""
import json,collections
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_subgoal_v1/development_step_001200')
counts=collections.Counter();lengths=[];mismatch=[];examples=[]
for path in sorted(root.glob('*/trace.json')):
    r=json.loads(path.read_text());counts['episodes']+=1;steps=r['steps']
    if steps and steps[-1]['action']==0:
        counts['stopped']+=1;lengths.append(len(steps));counts['immediate_stop']+=len(steps)==1
    for p in r['programs']:
        plan=p['plan'];counts['plans']+=1;counts['found']+=plan['found']
        distance=sum(a!=b for a,b in zip(p['initial_state'],plan['subgoal']));mismatch.append(distance)
        if plan['found']:
            counts['found_initial_equals_goal']+=distance==0
            counts['found_stop']+=plan['actions'][-1]==0
            counts['found_immediate_stop']+=plan['actions']==[0]
        if len(examples)<8:examples.append(dict(episode=r['episode']['episode_id'],initial=p['initial_state'],goal=plan['subgoal'],actions=plan['actions'],found=plan['found']))
result=dict(counts=counts,stop_step_counts=dict(collections.Counter(lengths)),initial_goal_differences=dict(collections.Counter(mismatch)),examples=examples)
output=root.parent/'subgoal_step1200_trace_diagnosis.json';output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
