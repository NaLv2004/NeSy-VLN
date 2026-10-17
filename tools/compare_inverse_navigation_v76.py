"""Read completed development traces; never generate another trajectory."""
import json,collections,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010')
jobs={'old_inverse':B/'perceptual_fsm_v74/tail_1e4/evaluations/step_000010',
 'observed_inverse':B/'observation_inverse_v76/observed_inverse/evaluations/step_000040',
 'no_inverse':B/'observation_inverse_v76/no_inverse/evaluations/step_000010'}
read=lambda p:json.loads(p.read_text())
output={'time':time.time(),'fresh_rollouts':0,'split':'development','arms':{}}
for arm,folder in jobs.items():
 summary=read(folder/'summary.json');assert summary['episodes']==80
 counts=collections.Counter();observed=[set() for _ in range(4)];memory=[set() for _ in range(4)]
 for path in sorted(folder.glob('*/trace.json')):
  trace=read(path);steps=trace['steps'];success=trace['metrics']['success'];counts['episodes']+=1
  counts['success']+=int(success);counts['false_stop']+=int(not success and steps[-1]['action']==0)
  counts['horizon_failure']+=int(not success and steps[-1]['action']!=0)
  for row in steps:
   for f in range(4):observed[f].add(row['observation_tokens'][f]);memory[f].add(row['state'][f])
  for previous,current in zip(steps,steps[1:]):
   counts['transitions']+=1;counts['repeated']+=int(previous['action']==current['action'])
   if previous['collision']:
    counts['postcollision_steps']+=1;counts['postcollision_repeated']+=int(previous['action']==current['action'])
 output['arms'][arm]={'directory':str(folder),'summary':summary,'counts':dict(counts),
  'observed_ids_used':list(map(len,observed)),'memory_argmax_ids_used':list(map(len,memory)),
  'repeat_fraction':counts['repeated']/counts['transitions'],
  'repeat_after_collision_fraction':counts['postcollision_repeated']/max(1,counts['postcollision_steps'])}
 assert counts['episodes']==80
target=B/'observation_inverse_v76/selected_development_behavior.json';target.write_text(json.dumps(output,indent=2))
for arm,x in output['arms'].items():x.pop('summary');print(arm,json.dumps(x))
