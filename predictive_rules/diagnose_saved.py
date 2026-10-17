"""Read-only diagnosis of completed development rollouts; no rerun or tuning."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from predictive_nesy.interpreter import Program, scalar, risk
from predictive_nesy import belief_interpreter as belief


def diagnose(directory):
    directory=Path(directory)
    actions=Counter();parents=Counter();goals=[];risks=[];collisions=[];confidence=[]
    unchanged=0;steps=0;repeat=0;long_plans=0;plans=0;failed_stops=0;stops=0
    predicted_arrivals=[]
    for path in directory.glob('*/trace.json'):
        trace=json.loads(path.read_text())
        programs={r['start']:Program.load(path.parent/r['file']) for r in trace['programs']}
        for r in trace['programs']:
            p=programs[r['start']];plans+=1
            parents.update(tuple(pair) for pair in p.parents.tolist())
            long_plans+=len(r['plan']['actions'])==10 and 0 not in r['plan']['actions']
            soft='beliefs' in r['plan']
            end=np.asarray(r['plan']['beliefs' if soft else 'states'][-1])
            predicted_arrivals.append(float((belief.scalar if soft else scalar)(p,end,'goals')))
        for i,s in enumerate(trace['steps']):
            p=programs[s['program_start']]
            action=s['action'];actions[action]+=1;steps+=1
            before=np.asarray(s['before']);after=np.asarray(s['after'])
            unchanged+=int(np.array_equal(before,after))
            repeat+=int(i>0 and action==trace['steps'][i-1]['action'])
            soft='before_belief' in s
            probs=np.asarray(s['after_belief'] if soft else s['transition_probabilities'])
            confidence.append(float(probs.max(-1).mean()))
            soft_before=np.asarray(s['before_belief']) if soft else before
            risks.append(float(belief.risk_all(p,soft_before)[action] if soft else risk(p,before,np.asarray(action))))
            collisions.append(s['collision_for_report_only'])
            goals.append(float((belief.scalar if soft else scalar)(p,soft_before,'goals')))
            if action==0:
                stops+=1;failed_stops+=not trace['metrics'].get('success',False)
    r=np.asarray(risks);c=np.asarray(collisions,dtype=float)
    report=dict(episodes=len(list(directory.glob('*/trace.json'))),steps=steps,action_histogram=dict(actions),
        repeated_action_rate=repeat/max(steps,1),unchanged_predicted_state_rate=unchanged/max(steps,1),
        most_common_parent_pairs=parents.most_common(),mean_transition_max_probability=float(np.mean(confidence)),
        predicted_collision_mean=float(r.mean()),actual_collision_mean=float(c.mean()),
        collision_brier=float(((r-c)**2).mean()),mean_goal_probability=float(np.mean(goals)),
        mean_planned_terminal_goal=float(np.mean(predicted_arrivals)),plans_without_stop=long_plans,
        programs=plans,stop_actions=stops,failed_stops=failed_stops)
    (directory/'diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evaluation');a=p.parse_args()
    print(json.dumps(diagnose(a.evaluation),indent=2))
