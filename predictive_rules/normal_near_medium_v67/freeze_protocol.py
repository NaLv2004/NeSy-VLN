"""Freeze the user-requested normal-layout near40/medium40 curriculum."""
import collections,hashlib,json,random,sys,time
from pathlib import Path
BASE=Path('/home/user/xjy/nesy_vla_20261010')
ROOT=BASE/'predictive_rules_20261015';OUT=ROOT/'normal_near_medium_v67'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(BASE/'symbolic_execution_v38/configs/grounded_outcome_feedback.json')
ref=Path(cfg['reference_root']);lock=read(ref/'research/split_lock.json');env=read(ref/'configs/experiment.json')
src=Path(env['source_data'])
assert all(sha(src/n)==lock['hashes'][n] for n in ('train.json','fixed80.json'))
all_tasks=read(src/'train.json');train_scenes=set(lock['train_scenes'])-set(lock['development_scenes'])
def eligible(e):return e['domain']=='normal' and e['distance_group'] in ('near','medium')
train=[e for e in all_tasks if e['scene'] in train_scenes and eligible(e)]
pool=[e for e in all_tasks if e['scene'] in lock['development_scenes'] and eligible(e)]
rng=random.Random(671017);dev=[]
for distance in ('near','medium'):
    for category in ('chair','table'):
        scenes=sorted({e['scene'] for e in pool})
        for i,scene in enumerate(scenes):
            candidates=sorted((e for e in pool if e['scene']==scene and e['category']==category and e['distance_group']==distance),key=lambda e:e['episode_id'])
            dev.extend(rng.sample(candidates,20//len(scenes)+(i<20%len(scenes))))
assert len(dev)==80 and len({e['episode_id'] for e in dev})==80
assert collections.Counter(e['distance_group'] for e in dev)=={'near':40,'medium':40}
assert not {e['scene'] for e in train}&{e['scene'] for e in dev}
assert not {e['scene'] for e in train+dev}&set(lock['fixed_scenes'])
OUT.mkdir(exist_ok=True)
for name,rows in [('train_tasks.json',train),('development80.json',dev)]:
    p=OUT/name
    if p.exists():assert read(p)==rows,'Frozen curriculum must not be replaced'
    else:p.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
source=read(ROOT/'counterfactual_closure_v63/manifest.json');allowed={e['episode_id'] for e in train}
boundary_ids=sorted(set(source['episode_ids'])&allowed)
def summary(rows):
    return {g:dict(count=len(v),minimum_m=min(v),maximum_m=max(v)) for g in ('near','medium') if (v:=[e['initial_distance_m'] for e in rows if e['distance_group']==g])}
manifest=dict(version='normal_near_medium_v67',created=time.time(),user_requested=True,
    train_tasks=len(train),train_summary=summary(train),development_tasks=80,development_summary=summary(dev),
    train_layouts=sorted({e['scene'] for e in train}),development_layouts=sorted({e['scene'] for e in dev}),
    development_category_counts=dict(collections.Counter(e['category'] for e in dev)),
    development_layout_counts=dict(collections.Counter(e['scene'] for e in dev)),
    source_hashes=lock['hashes'],train_sha256=sha(OUT/'train_tasks.json'),development_sha256=sha(OUT/'development80.json'),
    selected_boundary_episodes=boundary_ids,selected_boundary_episode_count=len(boundary_ids),
    excludes_dense=True,excludes_far=True,excludes_initial_stop_ready=True,max_actions=32,
    old_development_and_final_test_preserved=True,final_test_used=False,
    scope='All new optimization examples and runtime development tasks are normal near/medium; source checkpoint retains its historical pretraining',
    distance_metric='geodesic shortest navigable path to any recorded target viewpoint')
p=OUT/'manifest.json'
if p.exists():
    old=read(p);manifest['created']=old['created'];assert old==manifest
else:p.write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k not in ('selected_boundary_episodes','source_hashes')},indent=2))
