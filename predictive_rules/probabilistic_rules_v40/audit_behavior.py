from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]);rows=[json.loads(p.read_text()) for p in sorted(root.glob('*/trace.json'))]
assert len(rows)==80
actions=Counter();pairs=[];collisions=0;count=0;same_action=0;action_pairs=0;examples=[]
for row in rows:
    steps=row['steps']
    actions.update(s['action'] for s in steps);count+=len(steps)
    collisions+=sum(s['collision_for_report_only'] for s in steps)
    for left,right in zip(steps,steps[1:]):
        p=np.asarray(left['proof']['root_outcome_probabilities'])
        pairs.append(dict(changed=left['observed']!=right['observed'],correct=left['predicted_observation']==right['observed'],
            actual_probability=float(p[right['observed']]),maximum=float(p.max())))
        action_pairs+=1;same_action+=left['action']==right['action']
    if len(steps)>20 and len(examples)<5:
        examples.append(dict(episode=row['episode']['episode_id'],actions=[s['action'] for s in steps],
            observed=[s['observed'] for s in steps],memory=[s['memory'] for s in steps],metrics=row['metrics']))
changed=[p for p in pairs if p['changed']]
result=dict(episodes=80,actions=count,action_histogram=dict(actions),collision_rate=collisions/max(1,count),
    adjacent_same_action_rate=same_action/max(1,action_pairs),
    always_predict_current_observation_baseline=sum(not p['changed'] for p in pairs)/max(1,len(pairs)),
    changed_observation_accuracy=sum(p['correct'] for p in changed)/max(1,len(changed)),
    mean_maximum_observation_probability=float(np.mean([p['maximum'] for p in pairs])),examples=examples)
(root/'behavior_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
