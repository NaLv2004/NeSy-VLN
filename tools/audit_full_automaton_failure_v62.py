"""Read completed immutable development traces only; do not repeat inference."""
import collections,json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'full_automaton_rl_v62';target=out/'first40_failure_audit.json'
assert not target.exists()
result={'time':time.time(),'scope':'Existing completed development10/20/30/40 traces. No new navigation rollouts.','arms':{}}
for arm in ['joint','action_rules_only']:
 records=[]
 for step in [10,20,30,40]:
  folder=out/arm/'evaluations'/f'step_{step:06d}';summary=json.loads((folder/'summary.json').read_text());assert summary['episodes']==80
  traces=[json.loads(p.read_text()) for p in sorted(folder.glob('*/trace.json'))];assert len(traces)==80
  actions=collections.Counter();outcome=collections.Counter();distance_groups={};nearest=[];repeat=0;pairs=0;repeat_after_collision=0;collision_pairs=0;predicates=[set() for _ in range(4)]
  for t in traces:
   success=t['metrics'].get('success',False);stopped=t['steps'][-1]['action']==0
   outcome['success' if success else 'false_stop' if stopped else 'horizon_exhausted']+=1
   group=t['episode']['distance_group'];g=distance_groups.setdefault(group,dict(total=0,success=0,false_stop=0,horizon_exhausted=0));g['total']+=1
   g['success' if success else 'false_stop' if stopped else 'horizon_exhausted']+=1
   for row in t['steps']:
    actions[row['action']]+=1
    for i,symbol in enumerate(row['observation_tokens']):predicates[i].add(symbol)
   for a,z in zip(t['steps'],t['steps'][1:]):
    repeat+=int(a['action']==z['action']);pairs+=1
    if a['collision']:collision_pairs+=1;repeat_after_collision+=int(a['action']==z['action'])
  records.append({'update':step,'summary':summary,'outcomes':dict(outcome),'distance_groups':distance_groups,
   'action_histogram':dict(actions),'consecutive_repeat_fraction':repeat/max(1,pairs),
   'repeat_after_collision_fraction':repeat_after_collision/max(1,collision_pairs),'observation_ids_used_per_factor':[len(p) for p in predicates]})
 result['arms'][arm]=records
target.write_text(json.dumps(result,indent=2))
print(json.dumps({'time':result['time'],'arms':{a:[{k:v for k,v in r.items() if k in ['update','outcomes','consecutive_repeat_fraction','repeat_after_collision_fraction','observation_ids_used_per_factor']} for r in rows] for a,rows in result['arms'].items()}},indent=2))
