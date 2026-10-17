"""Measured training execution ambiguity before versus after symbolic compression."""
import collections,hashlib,json,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');R=Path('/home/user/xjy/strips_wm_navigation_20261009')
OUT=B/'training_alias_ceiling_v94';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
lock=read(R/'research/split_lock.json');cfg=read(R/'configs/experiment.json')
train_path=Path(cfg['source_data'])/'train.json'
assert hashlib.sha256(train_path.read_bytes()).hexdigest()==lock['hashes']['train.json']
allowed=set(lock['train_scenes'])-set(lock['development_scenes']);assert not allowed.intersection(lock['fixed_scenes'])
tasks={r['episode_id']:r for r in read(train_path)}
rows=[];seen=set();files=[];episodes=0
for update in range(61,71):
 paths=sorted((B/'parallel_ppo_v85/rollouts'/f'step_{update:06d}').glob('rank*/*.json'))
 assert len(paths)==192,(update,len(paths))
 for path in paths:
  content=path.read_bytes();t=json.loads(content);e=t['episode'];assert e==tasks[e['episode_id']] and e['scene'] in allowed
  episodes+=1;files.append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest()))
  for i,s in enumerate(t['steps']):
   if s['action']==0:continue
   # Keep one outcome per exact task-image-action per update. This avoids long
   # repeated collision loops counting as many independent visual examples.
   key=(update,e['category'],e['scene'],s['image_sha256'],s['action'])
   if key in seen:continue
   seen.add(key)
   before=t['steps'][i-1] if i else None
   rows.append(dict(update=update,category=e['category'],scene=e['scene'],episode=e['episode_id'],step=i,
    action=s['action'],collision=bool(s['collision']),image_hash=s['image_sha256'],codes=s['patch_tokens'],
    observed=s['observed_ids'],previous_action=s.get('previous_action',before['action'] if before else 17),
    state=s.get('state_ids'),previous_collision=bool(before['collision']) if before else False))

def measure(kind):
 groups=collections.defaultdict(list)
 for r in rows:
  prefix=(r['update'],r['category'],r['action'])
  if kind=='observed':key=prefix+tuple(r['observed'])
  elif kind=='patch_tokens':key=prefix+tuple(r['codes'])
  elif kind=='pixels':key=prefix+(r['image_hash'],)
  elif kind=='observed_previous_action':key=prefix+tuple(r['observed'])+(r['previous_action'],)
  groups[key].append(r)
 supported=[rs for rs in groups.values() if len(rs)>=2]
 errors=sum(min(sum(r['collision'] for r in rs),sum(not r['collision'] for r in rs)) for rs in groups.values())
 mixed=[rs for rs in supported if any(r['collision'] for r in rs) and not all(r['collision'] for r in rs)]
 return dict(groups=len(groups),singleton_groups=sum(len(rs)==1 for rs in groups.values()),
  supported_samples=sum(len(rs) for rs in supported),mixed_groups=len(mixed),mixed_samples=sum(len(rs) for rs in mixed),
  minimum_empirical_outcome_errors=errors,minimum_empirical_error_fraction=errors/len(rows),
  distinct_image_mixed_groups=sum(len({r['image_hash'] for r in rs})>1 for rs in mixed),
  largest_mixed_examples=[dict(n=len(rs),collision=sum(r['collision'] for r in rs),
   first={k:rs[0][k] for k in ['update','category','action','observed','episode','step']},
   unique_pixels=len({r['image_hash'] for r in rs}),unique_patch_grids=len({tuple(r['codes']) for r in rs})) for rs in sorted(mixed,key=len,reverse=True)[:10]])

result=dict(time=time.time(),source_run=str(B/'parallel_ppo_v85'),updates=[61,70],episodes=episodes,
 measured_examples=len(rows),collision_examples=sum(r['collision'] for r in rows),
 results={k:measure(k) for k in ['observed','observed_previous_action','patch_tokens','pixels']},
 files=files,train_hash=lock['hashes']['train.json'],allowed_layouts=sorted(allowed),
 qualification='In-sample deterministic label-conflict lower bound, not generalization accuracy. Keys include update so learned state dictionaries are not pooled across checkpoints. Singleton-rich full token/pixel groups have less opportunity to reveal ambiguity. No new rollouts or model updates; only completed original-training trajectories.')
(OUT/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
