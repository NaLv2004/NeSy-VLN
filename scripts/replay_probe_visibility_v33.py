"""Train-only simulator audit of observation ambiguity; no policy or dataset edits."""
import collections,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='7'
import numpy as np
from PIL import Image
base=Path('/home/user/xjy/nesy_vla_20261010');ref=Path('/home/user/xjy/strips_wm_navigation_20261009');sys.path.insert(0,str(ref))
from stripsnav.io import read_json,atomic_json,sha256
from stripsnav.sim_adapter import NavigationEnvironment
out=base/'visibility_replay_v33';out.mkdir(exist_ok=True)
lockfile=(out/'lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=read_json(ref/'configs/experiment.json');cfg.update(training_horizon=32,evaluation_horizon=32)
lock=read_json(ref/'research/split_lock.json');allowed=set(lock['train_scenes'])-set(lock['development_scenes'])
tasks={e['episode_id']:e for e in read_json(Path(cfg['source_data'])/'train.json')}
assert sha256(Path(cfg['source_data'])/'train.json')==lock['hashes']['train.json']
dataset=read_json(base/'conjunction_learning_v31/dataset.json')
demo_root=base/'data/lossless_demonstrations_v7';rows=read_json(demo_root/'examples.json');first={}
for row in rows:
    eid=row['episode_id']
    if eid not in first or row['image']<first[eid]['image']:first[eid]={**row,'_root':str(demo_root)}
paired_root=base/'data/paired_goals_v12'
for pair in read_json(paired_root/'pairs.json'):
    if not pair['both_successful'] or not pair['same_initial_rgb']:continue
    assert pair['scene'] in allowed
    for branch in pair['branches']:
        episode=branch['episode'];eid=episode['episode_id']
        assert episode['scene']==pair['scene'] and eid not in tasks
        tasks[eid]=episode;first[eid]={**branch['frames'][0],'_root':str(paired_root)}
selected=sorted(dataset['examples'],key=lambda e:(e['scene'],e['id']))
assert all(e['scene'] in allowed and tasks[e['id']]['scene']==e['scene'] for e in selected)
free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>3500,'Do not compete with training for the last VRAM'
state={'state':'running','pid':os.getpid(),'time':time.time(),'gpu':7,'total':len(selected),'rows':[],
    'train_only':True,'original_fixed_or_dev_used':False,'visibility_is_diagnostic_only':True}
atomic_json(out/'status.json',state)
env=NavigationEnvironment(cfg,gpu=0)
try:
    for e in selected:
        path=out/(e['id']+'.json')
        if path.exists():record=read_json(path)
        else:
            image_path=Path(first[e['id']]['_root'])/first[e['id']]['image']
            raw=env.reset(tasks[e['id']]);reference=np.asarray(Image.open(image_path).convert('RGB'))
            actual=np.asarray(raw)[...,:3];assert actual.shape==reference.shape
            error=np.abs(actual.astype(np.int16)-reference.astype(np.int16))
            metric=env.private_metrics();record={k:e[k] for k in ['id','scene','category','label','heldout']}
            record.update(visible_pixels=metric['visible_pixels'],eligible=metric['eligible'],
                rgb_mae=float(error.mean()),rgb_max_difference=int(error.max()),rgb_identical=bool((error==0).all()),
                source_image=str(image_path),source_sha256=sha256(image_path))
            if error.mean()>.05:raise RuntimeError('Replay pixels no longer match frozen training images')
            atomic_json(path,record)
        state['rows'].append(record)
        if len(state['rows'])%20==0:
            state['time']=time.time();atomic_json(out/'status.json',state)
    for part in ['all','training','probe_holdout']:
        current=[r for r in state['rows'] if part=='all' or r['heldout']==(part=='probe_holdout')]
        groups={}
        for visible in [True,False]:
            group=[r for r in current if (r['visible_pixels']>=16)==visible]
            groups['visible' if visible else 'not_visible']={'frames':len(group),
                'actions':dict(collections.Counter(r['label'] for r in group)),
                'eligible':sum(r['eligible'] for r in group)}
        state.setdefault('summary',{})[part]=groups
    state.update(state='complete',time=time.time());atomic_json(out/'status.json',state)
    print(json.dumps({'state':'complete','summary':state['summary'],
        'max_rgb_mae':max(r['rgb_mae'] for r in state['rows'])},indent=2),flush=True)
except BaseException as error:
    state.update(state='failed',time=time.time(),error=repr(error));atomic_json(out/'status.json',state);raise
finally:env.close()
