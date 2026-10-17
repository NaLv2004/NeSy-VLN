"""Read completed validation traces; measure, never feed them into training."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('evaluation_directory');args=parser.parse_args()
root=Path(args.evaluation_directory)
records=[json.loads(p.read_text()) for p in sorted(root.glob('*/trace.json'))]
assert len(records)==80,'Only a complete unchanged dev80 may be summarized'
labels=[];scores=[];nll=[];top1=[];top4=[];briers=[];proof_failures=0
for record in records:
    success=bool(record.get('metrics',{}).get('success',False))
    steps=record['steps'];proof_failures+=bool(record.get('planning_failed'))
    labels.append(int(success));scores.append(steps[0]['proof']['model_goal_probability'] if steps else 0.)
    for before,after in zip(steps,steps[1:]):
        p=np.asarray(before['proof']['root_outcome_probabilities'],dtype=np.float64)
        target=int(after['observed']);assert abs(float(p.sum())-1)<2e-6
        nll.append(-math.log(max(float(p[target]),1e-12)))
        top1.append(int(p.argmax())==target);top4.append(target in p.argsort()[-4:])
        true=np.zeros_like(p);true[target]=1;briers.append(float(np.square(p-true).sum()))
bins=[]
for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.0000001)]:
    ids=[i for i,s in enumerate(scores) if lo<=s<hi]
    bins.append(dict(lower=lo,upper=min(hi,1),episodes=len(ids),
        predicted_goal_probability=float(np.mean([scores[i] for i in ids])) if ids else None,
        actual_success_rate=float(np.mean([labels[i] for i in ids])) if ids else None))
result=dict(episodes=len(records),successes=sum(labels),no_proof=proof_failures,
    predicted_goal_probability_mean=float(np.mean(scores)),
    observed_transition_count=len(nll),observation_nll=float(np.mean(nll)) if nll else None,
    observation_top1=float(np.mean(top1)) if top1 else None,
    observation_top4=float(np.mean(top4)) if top4 else None,
    observation_brier=float(np.mean(briers)) if briers else None,calibration_bins=bins,
    note='Model goal probability includes perception and rule assumptions; it is not measured physical success. No gradients or planning changes use these validation records.')
output=root/'probability_calibration.json';output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
