"""Audit completed v43 proofs; counterfactuals are not navigation evaluations."""
import json,sys,time,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
b=Path('/home/user/xjy/nesy_vla_20261010');root=b/'spatial_automaton_v43'
sys.path.insert(0,str(b/'symbolic_paired_v12'))
from nesy_vla.token_rule_revision import TokenFacts,convert_v43_trees,PackedRuleProgram
read=lambda p:json.loads(p.read_text());schema=(64,)*49+(65,)*50
assert read(root/'status.json')['state']=='complete'
raw=read(root/'programs.json');programs={k:PackedRuleProgram(convert_v43_trees(t,schema),schema) for k,t in raw.items()}
records=[];task_rows=defaultdict(list)
for marker in sorted((root/'development').glob('*/evaluation.json')):
 trace=read(marker.parent/'trace.json');episode=read(marker);steps=trace['steps'];actions=[s['action'] for s in steps]
 repeated=sum(a==z for a,z in zip(actions[1:],actions));aftercollision=sum(s['collision'] for s in steps[:-1]);collrepeat=sum(a['collision'] and a['action']==z['action'] for a,z in zip(steps,steps[1:]))
 code_same=sum(a['symbols'][:49]==z['symbols'][:49] for a,z in zip(steps,steps[1:]));image_same=sum(a['image_sha256']==z['image_sha256'] for a,z in zip(steps,steps[1:]))
 p=np.array([s['probabilities'] for s in steps]);sort=np.sort(p,axis=1)
 seen={};repeatstates=0;longest=0;current=0
 for i,s in enumerate(steps):
  token=tuple(s['symbols']);repeatstates+=int(token in seen);seen[token]=i
  current=current+1 if s['collision'] else 0;longest=max(longest,current)
  task_rows[episode['category']].append((episode['episode_id'],s))
 records.append(dict(episode,action_counts=dict(Counter(actions)),repeated_actions=repeated,consecutive_pairs=len(steps)-1,after_collision_pairs=aftercollision,repeated_after_collision=collrepeat,identical_current_codes=code_same,identical_images=image_same,revisited_full_states=repeatstates,longest_collision_run=longest,final_distance=trace['metrics'].get('distance'),final_visible_pixels=trace['metrics'].get('visible_pixels'),last_action=actions[-1],chosen_probability_mean=float(p.max(1).mean()),top2_margin_mean=float((sort[:,-1]-sort[:,-2]).mean())))
def aggregate(rows):
 keys=['steps','collisions','repeated_actions','consecutive_pairs','after_collision_pairs','repeated_after_collision','identical_current_codes','identical_images','revisited_full_states']
 result={k:sum(r[k] for r in rows) for k in keys};result.update(episodes=len(rows),successes=sum(r['success'] for r in rows),stop_episodes=sum(r['last_action']==0 for r in rows),mean_chosen_probability=float(np.mean([r['chosen_probability_mean'] for r in rows])),collision_rate=result['collisions']/max(1,result['steps']))
 return result
ablation={}
for task,rows in task_rows.items():
 states=np.array([s['symbols'] for _,s in rows],dtype=np.int64);p=np.array([s['probabilities'] for _,s in rows]);original=p.argmax(1)
 def probabilities(x):
  results=[]
  for start in range(0,len(x),128):
   ev,_=programs[task].evidence(TokenFacts(torch.from_numpy(x[start:start+128]),schema));results.append(ev.mean(1))
  return np.concatenate(results)
 assert np.max(np.abs(probabilities(states)-p))<1e-10
 variants={};x=states.copy();x[:,-1]=64;variants['unknown_previous_action']=x
 x=states.copy();x[:,49:98]=64;variants['unknown_previous_image']=x
 x=states.copy();x[:,:49]=np.roll(states[:,:49],1,axis=0);variants['different_observed_current_image_tokens']=x
 variants={name:{'changed_actions':int(np.sum(probabilities(x).argmax(1)!=original)),'steps':len(x)} for name,x in variants.items()}
 switched=np.zeros(len(states),dtype=bool)
 for action in range(17):
  x=states.copy();x[:,-1]=action;switched|=probabilities(x).argmax(1)!=original
 literal=Counter();root_literal=Counter()
 for _,s in rows:
  for proof in s['proof']:
   for depth,(slot,token,truth) in enumerate(proof['clauses']):
    group='current_visual_token' if slot<49 else ('previous_visual_token' if slot<98 else 'previous_action')
    literal[group]+=1
    if depth==0:root_literal[group]+=1
 ablation[task]={'variants':variants,'some_previous_action_changes_decision':int(switched.sum()),'steps':len(states),'executed_literal_counts':dict(literal),'root_literal_counts':dict(root_literal)}
result={'time':time.time(),'program_commit':read(root/'COMMITTED'),'scope':'Completed original80 development traces only; no new rollout, training or fixed80 selection. Token substitutions are diagnostic interventions, not physically valid trajectories or performance estimates.','all':aggregate(records),'success':aggregate([r for r in records if r['success']]),'failure':aggregate([r for r in records if not r['success']]),'ablation':ablation,'episodes':records}
(root/'failure_audit_v46.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
