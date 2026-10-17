"""Read immutable v2 trajectories only; diagnose numerical/program collapse."""
import collections,json,statistics
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/v2/runs/full_v2')
out={}
for update in (10,20):
    folder=root/f'evaluations/step_{update:06d}/development'
    beliefs=[];confidences=[];weights=[];supports=[];errors=collections.Counter();actions=collections.Counter()
    for file in folder.glob('*/trajectory.json'):
        trace=json.loads(file.read_text());error=trace.get('metrics',{}).get('inference_failure')
        if error:errors[error]+=1
        for row in trace['steps']:
            beliefs.extend(row['observed_truth']);confidences.extend(row['confidence'])
            supports.append(sum(row['support']));actions[row['action']]+=1
        for program in file.parent.glob('program_*.json'):
            p=json.loads(program.read_text())['program'];r=len(p['weights']);weights.extend(p['weights'][r//3:2*r//3])
    def stats(values):
        return {'min':min(values),'max':max(values),'mean':statistics.mean(values),
                'below_1e-5':sum(v<1e-5 for v in values)/len(values),
                'above_1_minus_1e-5':sum(v>1-1e-5 for v in values)/len(values)} if values else None
    out[str(update)]={'belief':stats(beliefs),'confidence':stats(confidences),'action_weights':stats(weights),
                     'support_sum':stats(supports),'errors':dict(errors),'actions':dict(actions)}
(root/'symbolic_health/saturation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
