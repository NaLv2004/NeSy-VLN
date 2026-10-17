"""Analyze completed saved navigation traces, never rerun an episode."""
import json,time,statistics
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/perceptual_fsm_v74')
read=lambda p:json.loads(p.read_text())
out=b/'behavior_audit_10_30.json'
assert not out.exists(),'Completed trace audit already exists'
result={'time':time.time(),'fresh_rollouts':0,'runs':{}}
for arm in ['frozen','tail_1e4']:
 for epoch in [10,20,30]:
  root=b/arm/'evaluations'/f'step_{epoch:06d}';summary=read(root/'summary.json');assert summary['episodes']==80
  traces=[read(p) for p in sorted(root.glob('*/trace.json'))];assert len(traces)==80
  counts={'success':0,'false_stop':0,'horizon':0,'actions':0,'collisions':0,'adjacent':0,'repeated':0,'after_collision':0,'repeat_after_collision':0,'early_false_stop_3':0}
  observed=[set() for _ in range(4)];actions={};examples=[]
  for t in traces:
   steps=t['steps'];success=bool(t['metrics'].get('success'));stopped=steps[-1]['action']==0
   failure='success' if success else 'false_stop' if stopped else 'horizon';counts[failure]+=1
   counts['actions']+=len(steps);counts['collisions']+=sum(x['collision'] for x in steps)
   if failure=='false_stop' and len(steps)<=3:counts['early_false_stop_3']+=1
   for row in steps:
    actions[str(row['action'])]=actions.get(str(row['action']),0)+1
    for i,v in enumerate(row['observation_tokens']):observed[i].add(v)
   for old,new in zip(steps,steps[1:]):
    counts['adjacent']+=1;counts['repeated']+=old['action']==new['action']
    if old['collision']:counts['after_collision']+=1;counts['repeat_after_collision']+=old['action']==new['action']
   if not success and len(examples)<6:examples.append({'episode':t['episode']['episode_id'],'failure':failure,'metrics':t['metrics'],'action_sequence':[x['action'] for x in steps]})
  result['runs'][f'{arm}/{epoch}']={'summary':summary,'counts':counts,'action_counts':actions,
   'observed_ids_used':list(map(len,observed)),'repeat_rate':counts['repeated']/max(1,counts['adjacent']),
   'repeat_after_collision_rate':counts['repeat_after_collision']/max(1,counts['after_collision']),
   'failure_examples':examples}
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:{a:v[a] for a in ['counts','observed_ids_used','repeat_rate','repeat_after_collision_rate']} for k,v in result['runs'].items()},indent=2))
