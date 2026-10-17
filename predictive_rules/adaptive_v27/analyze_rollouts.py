"""Read-only diagnosis of saved hard proofs and renewal decisions.

No new rollout, no policy change, and no development gradients are performed.
This distinguishes goal reachability failures, dynamics mismatch and renewals.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np


def fraction(yes,total):return yes/total if total else None


def analyze(folder):
    folder=Path(folder);records=[];counts=Counter();actions=Counter();observed=Counter();corrected=Counter();targets=Counter()
    probabilities=[];gaps=[];categories=defaultdict(Counter);examples=[]
    for path in sorted(folder.glob('*/trace.json')):
        record=json.loads(path.read_text());records.append(record)
        steps=record['steps'];programs=record['programs'];task=record['episode']
        success=bool(record['metrics'].get('success',False)) and not record['planning_failed']
        counts['episodes']+=1;counts['successes']+=success;counts['executed_actions']+=len(steps)
        counts['programs']+=len(programs);counts['renewals']+=max(0,len(programs)-1)
        counts['initial_goal_unreachable']+=record['planning_failed'] and not steps
        counts['later_goal_unreachable']+=record['planning_failed'] and bool(steps)
        counts['failed_stop']+=not success and bool(steps) and steps[-1]['action']==0
        counts['action_limit']+=not success and len(steps)==32
        counts['episodes_never_renewed']+=len(programs)==1
        for before,after in zip(programs,programs[1:]):gaps.append(after['start']-before['start'])
        for event in record['renewal_decisions']:
            if event['step']>0:probabilities.append(event['probability'])
        for i,step in enumerate(steps):
            proof=step['proof'];actions[step['action']]+=1;observed[step['observed_id']]+=1
            corrected[proof['corrected']]+=1;targets[proof['goal']]+=1
            counts['correction_equals_observation']+=proof['corrected']==step['observed_id']
            counts['correction_equals_prediction']+=proof['corrected']==proof['predicted']
            counts['current_equals_generated_goal']+=proof['corrected']==proof['goal']
            counts['collisions']+=step['collision_for_report_only']
            if i and steps[i-1]['program_sha256']==step['program_sha256']:
                previous=steps[i-1];counts['comparable_transitions']+=1
                counts['predicted_next_matches_observation']+=previous['proof']['states'][1]==step['observed_id']
                counts['unchanged_observation']+=previous['observed_id']==step['observed_id']
                counts['repeated_action']+=previous['action']==step['action']
        group=':'.join(str(task.get(key,'')) for key in ('domain','distance_group','category'))
        categories[group]['episodes']+=1;categories[group]['successes']+=success
        categories[group]['initial_unreachable']+=record['planning_failed'] and not steps
        if record['planning_failed'] and len(examples)<12:
            failed=record['failed_proof']
            examples.append(dict(episode=task['episode_id'],executed=len(steps),programs=len(programs),
                                 last_renewal_probability=record['renewal_decisions'][-1]['probability'],
                                 failed_observation=failed['observed'],failed_prediction=failed['predicted'],
                                 failed_corrected=failed['corrected'],goal=failed['goal']))
    n=counts['executed_actions'];t=counts['comparable_transitions']
    return dict(complete_evaluation=(folder/'summary.json').exists(),counts=dict(counts),
        success_rate=fraction(counts['successes'],counts['episodes']),
        actions_per_program=fraction(n,counts['programs']),
        observation_preserved_rate=fraction(counts['correction_equals_observation'],n),
        predicted_next_observation_accuracy=fraction(counts['predicted_next_matches_observation'],t),
        repeated_action_rate=fraction(counts['repeated_action'],t),
        unchanged_observation_rate=fraction(counts['unchanged_observation'],t),
        collision_rate=fraction(counts['collisions'],n),
        renewal_probability=dict(mean=float(np.mean(probabilities)),maximum=max(probabilities),
            p50=float(np.quantile(probabilities,.5)),p95=float(np.quantile(probabilities,.95))) if probabilities else None,
        renewal_gaps=gaps,action_histogram=dict(actions),observation_unique=len(observed),corrected_unique=len(corrected),
        top_observed=observed.most_common(12),top_corrected=corrected.most_common(12),top_goals=targets.most_common(12),
        by_group={key:dict(value) for key,value in categories.items()},unreachable_examples=examples)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--out');args=p.parse_args()
    result=analyze(args.folder);text=json.dumps(result,indent=2,allow_nan=False)
    if args.out:Path(args.out).write_text(text)
    print(text)
