"""Describe existing rollouts; never rerun or alter a policy or validation task."""
import argparse,json,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True);a=p.parse_args()
folder=Path(a.evaluation);rows=[]
for trace in sorted(folder.glob('*/trace.json')):
    r=json.loads(trace.read_text());steps=r['steps'];programs=r['programs'];initial=programs[0];plan=initial['plan']
    last=steps[-1] if steps else None
    with np.load(trace.parent/initial['file'],allow_pickle=False) as data:
        state=int(initial['initial_state'][0]);goal=int(data['subgoal'][0]);rules=data['transitions'].argmax(-1)
        stop_next=int(rules[state,0])
    evaluation=json.loads((trace.parent/'evaluation.json').read_text())
    reason=('success' if evaluation['success'] else 'unreachable_generated_subgoal' if r['planning_failed'] else
            'premature_STOP' if last is not None and last['action']==0 else 'action_budget_exhausted')
    rows.append(dict(episode_id=trace.parent.name,success=evaluation['success'],reason=reason,steps=len(steps),
        initial_state=state,generated_subgoal=goal,initial_equals_subgoal=state==goal,
        initial_stop_successor=stop_next,initial_stop_changes_state=state!=stop_next,
        first_plan=plan['actions'],first_plan_length=len(plan['actions']),
        final_stop_changes_state=bool(last is not None and last['action']==0 and last['before']!=last['after']),
        actions=[s['action'] for s in steps],category=r['episode']['category'],
        distance_group=r['episode']['distance_group'],domain=r['episode']['domain'],metrics=r['metrics']))
stats=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
    failure_reasons=dict(collections.Counter(r['reason'] for r in rows)),
    first_plan_length_counts=dict(collections.Counter(r['first_plan_length'] for r in rows)),
    initial_equals_subgoal=sum(r['initial_equals_subgoal'] for r in rows),
    stop_changes_state=sum(r['final_stop_changes_state'] for r in rows),
    failed_stop_changes_state=sum(r['final_stop_changes_state'] and not r['success'] for r in rows),
    repeated_movement_episodes=sum(len(r['actions'])>=3 and len(set(x for x in r['actions'] if x!=0))==1 for r in rows))
result=dict(evaluation=str(folder),read_only_existing_traces=True,**stats,rows=rows)
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(stats))
