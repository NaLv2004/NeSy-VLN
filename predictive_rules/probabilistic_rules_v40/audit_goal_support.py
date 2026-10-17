"""Immutable validation diagnostic; no model, optimization or simulator calls."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('evaluation_directory');args=parser.parse_args()
root=Path(args.evaluation_directory);records=[]
for path in sorted(root.glob('*/trace.json')):
    trace=json.loads(path.read_text())
    for pr in trace['programs']:
        with np.load(path.parent/pr['file']) as p:
            m=p['next_memory'];probs=p['outcome_probability'];goal=int(p['goal']);nobs=probs.shape[-1]
            gm,go=divmod(goal,nobs)
            preimage=np.flatnonzero((m[:,0]==gm)&(probs[:,0,go]>0))
            records.append(dict(episode=trace['episode']['episode_id'],start=pr['start'],goal=goal,
                goal_memory=gm,goal_observation=go,preimage_count=len(preimage),
                any_stop_preimage=bool(len(preimage)),
                stopped_before_next_program=trace.get('planning_failed',False) and pr==trace['programs'][-1],
                actual_steps=sum(pr['start']<=s['step']<(trace['programs'][trace['programs'].index(pr)+1]['start'] if pr!=trace['programs'][-1] else 32) for s in trace['steps'])))
assert len({r['episode'] for r in records})==80
failed=[r for r in records if r['stopped_before_next_program']]
result=dict(episodes=80,programs=len(records),unsupported_generated_goals=sum(not r['any_stop_preimage'] for r in records),
    failed_programs=len(failed),failed_with_no_stop_preimage=sum(not r['any_stop_preimage'] for r in failed),
    program_rows=records,
    qualification='An empirical model may still be wrong when a symbolic preimage exists. This checks structural goal compatibility, not physical reachability.')
(root/'goal_support_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='program_rows'}))
