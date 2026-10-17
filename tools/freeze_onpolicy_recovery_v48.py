"""Freeze train-only on-policy prefixes; never collect held-out corrections."""
import collections,hashlib,json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');source=b/'token_revision_rl_v45';out=b/'data/onpolicy_recovery_v48';out.mkdir(parents=True,exist_ok=True)
assert not (out/'manifest.json').exists(),'Inspect existing collector before any resume'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(source/'manifest.json');allowed=set(manifest['data']['train_layouts']);excluded=set(manifest['data']['development_layouts'])|set(manifest['data']['fixed_layouts']);assert not allowed&excluded
assert (source/'step_000010/COMMITTED').exists() and (source/'evaluations/step_000010/summary.json').exists()
items=[];seen=set();groups=collections.Counter();source_success=0
for update in range(1,11):
 paths=sorted((source/'rollouts'/f'step_{update:06d}').glob('*.json'));assert len(paths)==16
 for path in paths:
  r=read(path);ep=r['episode'];steps=r['steps'];assert ep['scene'] in allowed and ep['scene'] not in excluded
  if r['metrics']['success']:source_success+=1;continue
  # Use the first actually encountered collision, including its resulting image.
  # If no collision occurred, retain the failed rollout prefix before STOP.
  candidate=next((i+1 for i,s in enumerate(steps) if s['collision']),None)
  if candidate is None:candidate=max(0,len(steps)-int(steps[-1]['action']==0))
  prefix_length=min(candidate,24);prefix=[s['action'] for s in steps[:prefix_length]]
  if 0 in prefix:prefix=prefix[:prefix.index(0)];prefix_length=len(prefix)
  token=(ep['episode_id'],tuple(prefix))
  if token in seen:continue
  seen.add(token);key=hashlib.sha256((ep['episode_id']+json.dumps(prefix)).encode()).hexdigest()[:16]
  item={'id':'R48_'+key,'episode':ep,'source_rollout':str(path),'source_rollout_sha256':sha(path),'prefix_actions':prefix,'prefix_length':prefix_length,'source_update':update,'selection':'first_observed_collision' if candidate is not None and any(s['collision'] for s in steps) else 'failed_noncollision_prefix'}
  items.append(item);groups[(ep['domain'],ep['category'],ep['distance_group'])]+=1
items.sort(key=lambda r:(r['episode']['scene'],r['id']))
result={'time':time.time(),'source_program_commit':manifest['source_commit'],'source_updates':list(range(1,11)),'source_episodes':160,'source_successes':source_success,'attempts':len(items),'groups':{'/'.join(k):v for k,v in groups.items()},'items':items,'source_data_hashes':{k:manifest['data'][k] for k in ['train_hash','fixed_hash']},'train_layouts':sorted(allowed),'excluded_layouts':sorted(excluded),'training_horizon':32,'fixed_or_development_used':False,'human_predicate_labels':False,'teacher':'Existing private simulator oracle used only to generate training action labels; no geometric inputs, scene map, or oracle is passed to learned policy. Only successful full trajectories contribute imitation targets; prefix actions remain context only.'}
(out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['items','source_program_commit']},indent=2))
