"""Audit completed navigation traces without new rollouts or controller feedback."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from predictive_nesy.graph_subgoal_interpreter import Program,plan

def inspect(folder):
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['episodes']==80, 'Only a completed fixed development evaluation can be audited'
    rows=[]
    for file in sorted(folder.glob('*/trace.json')):
        trace=json.loads(file.read_text());evaluation=json.loads((file.parent/'evaluation.json').read_text())
        programs=[]
        for event in trace['programs']:
            p=Program.load(file.parent/event['file']);saved=event['plan']
            recomputed=plan(p,event['initial_state'],horizon=min(10,32-event['start']))
            assert recomputed['found']==saved['found'] and recomputed['actions']==saved['actions']
            assert recomputed['states']==saved['states'] and recomputed['subgoal']==saved['subgoal']
            rules=p.transitions.argmax(-1);s=int(event['initial_state'][0]);g=int(p.subgoal[0])
            programs.append(dict(start=event['start'],found=saved['found'],initial=s,goal=g,
                initial_equals_goal=s==g,planned_actions=saved['actions'],planned_states=saved['states'],
                stop_can_reach_generated_goal=int(rules[s,0])==g))
        actions=[s['action'] for s in trace['steps']];movement=[a for a in actions if a!=0]
        stopped=bool(actions and actions[-1]==0)
        last=trace['steps'][-1] if stopped else None
        rows.append(dict(episode_id=evaluation['episode_id'],success=evaluation['success'],
            domain=evaluation['domain'],distance_group=evaluation['distance_group'],category=evaluation['category'],
            planning_failed=evaluation['planning_failed'],steps=len(actions),stopped=stopped,
            failed_after_stop=stopped and not evaluation['success'],
            stop_changes_symbol=bool(stopped and last['before']!=last['after']),
            executed_self_transitions=sum(s['before']==s['after'] for s in trace['steps'] if s['action']),
            movement_actions=len(movement),single_repeated_movement=bool(len(movement)>1 and len(set(movement))==1),
            repeated_adjacent_movement=sum(x==y for x,y in zip(movement,movement[1:])),
            collisions=evaluation['collisions'],initial_equals_goal=programs[0]['initial_equals_goal'],programs=programs))
    assert len(rows)==80
    counts=Counter()
    for r in rows:
        for k in ['success','planning_failed','stopped','failed_after_stop','stop_changes_symbol',
                  'executed_self_transitions','movement_actions','single_repeated_movement',
                  'repeated_adjacent_movement','collisions','initial_equals_goal']:
            counts[k]+=r[k]
    return dict(evaluation=str(folder),update=summary.get('update'),episodes=len(rows),
        completed_trace_rederivations_exact=True,no_new_rollouts=True,counts=dict(counts),rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=inspect(Path(a.evaluation));Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
