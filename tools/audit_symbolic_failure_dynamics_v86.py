"""Audit archived development trajectories only. No inference or rollouts."""
import json,math,time
from collections import Counter,defaultdict
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');source=B/'parallel_ppo_v84/evaluations/step_000040'
read=lambda p:json.loads(p.read_text())
assert read(source/'summary.json')['episodes']==80
def tv(a,b):return sum(abs(x-y) for rowa,rowb in zip(a,b) for x,y in zip(rowa,rowb))/8
def observation(s):return tuple(s['observation_tokens'])
def memory(s):return tuple(s['state'])
rows=[];totals=defaultdict(Counter);stop_alias=defaultdict(Counter)
for path in sorted(source.glob('*/trace.json')):
 t=read(path);e=t['episode'];steps=t['steps'];success=t['metrics']['success']
 kind='success' if success else ('false_stop' if steps[-1]['action']==0 else 'horizon')
 c=totals[kind];c['episodes']+=1;c['actions']+=len(steps)
 record={'episode_id':e['episode_id'],'category':e['category'],'domain':e['domain'],'difficulty':e['distance_group'],
  'initial_distance':e['initial_distance_m'],'outcome':kind,'actions':len(steps),'collisions':sum(s['collision'] for s in steps),
  'final_distance':t['metrics']['distance'],'final_visible_pixels':t['metrics']['visible_pixels'],
  'final_stop_probability':steps[-1]['probabilities'][0]}
 run=0;longest=0;loop_start=None;loop_blocks=[]
 for i,s in enumerate(steps):
  if s['collision']:c['collisions']+=1
  if i==0:continue
  prev=steps[i-1];repeated=s['action']==prev['action'];unchanged=observation(s)==observation(prev)
  if prev['collision']:
   c['after_collision']+=1;c['same_action_after_collision']+=repeated
   c['same_observation_after_collision']+=unchanged
   c['same_patch_grid_after_collision']+=s['patch_tokens']==prev['patch_tokens']
   c['same_memory_argmax_after_collision']+=memory(s)==memory(prev)
   c['near_stationary_belief_after_collision']+=tv(s['state_belief'],prev['state_belief'])<.01
   c['same_program_after_collision']+=s['program_start']==prev['program_start']
  continuing=prev['collision'] and s['collision'] and repeated
  if continuing:
   if run==0:loop_start=i-1;run=1
   run+=1
  else:
   if run>=4:loop_blocks.append((loop_start,i-1,run))
   longest=max(longest,run);run=0
 if run>=4:loop_blocks.append((loop_start,len(steps)-1,run))
 longest=max(longest,run);record['longest_same_action_collision_run']=longest;record['collision_loop_blocks']=loop_blocks
 if longest>=4:c['episodes_with_4plus_collision_loop']+=1
 if kind=='false_stop':
  d=record['final_distance'];v=record['final_visible_pixels']
  c['distance_le_1']+=d<=1;c['distance_1_to_2']+=1<d<=2;c['distance_2_to_5']+=2<d<=5;c['distance_gt_5']+=d>5
  c['target_not_visible']+=v<16
 if steps[-1]['action']==0:
  stop_alias[(e['category'],observation(steps[-1]))][kind]+=1
  record['final_observation_ids']=list(observation(steps[-1]));record['final_memory_ids']=list(memory(steps[-1]))
 rows.append(record)
overlaps=[{'category':k[0],'observed_ids':k[1],'outcomes':dict(v)} for k,v in stop_alias.items() if v['success'] and v['false_stop']]
result={'time':time.time(),'source':str(source),'summary':read(source/'summary.json'),'totals':{k:dict(v) for k,v in totals.items()},
 'stop_observation_aliases':overlaps,'rows':rows,
 'qualification':'Observed-ID aliases alone do not prove complete policy-state aliasing: soft memory and generated programs may differ. No hand-assigned state meanings. No original fixed80 examined.'}
out=B/'failure_dynamics_v86';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','summary']},indent=2))
print(json.dumps({'worst_loops':sorted(rows,key=lambda r:r['longest_same_action_collision_run'],reverse=True)[:6]},indent=2))
