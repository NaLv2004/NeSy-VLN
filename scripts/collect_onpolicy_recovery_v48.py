"""Training-only recovery collection from hash-verified on-policy failure prefixes."""
import argparse,fcntl,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
base=Path('/home/user/xjy/nesy_vla_20261010');ref=Path('/home/user/xjy/strips_wm_navigation_20261009');sys.path.insert(0,str(ref))
from stripsnav.io import atomic_json,read_json,sha256
from stripsnav.sim_adapter import NavigationEnvironment
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--world-size',type=int,default=6);args=p.parse_args()
out=base/'data/onpolicy_recovery_v48';manifest=read_json(out/'manifest.json');run=out/f'rank_{args.rank}';run.mkdir(exist_ok=True)
lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=read_json(ref/'configs/experiment.json');cfg.update(training_horizon=32,evaluation_horizon=32)
source=Path(cfg['source_data']);assert sha256(source/'train.json')==manifest['source_data_hashes']['train_hash'] and sha256(source/'fixed80.json')==manifest['source_data_hashes']['fixed_hash']
allowed=set(manifest['train_layouts']);excluded=set(manifest['excluded_layouts']);assert not allowed&excluded
selected=manifest['items'][args.rank::args.world_size];state={'pid':os.getpid(),'state':'initializing','time':time.time(),'attempts':len(selected)}
def progress(**fields):state.update(fields,time=time.time());atomic_json(run/'status.json',state);print(json.dumps(state),flush=True)
def pixel_hash(rgb):return hashlib.sha256(np.asarray(rgb)[...,:3].astype(np.uint8).tobytes()).hexdigest()
env=None
try:
 env=NavigationEnvironment(cfg,gpu=0,evaluation=True);done=[]
 for item in selected:
  folder=run/item['id'];marker=folder/'COMMITTED'
  if marker.exists():done.append(read_json(marker));continue
  episode=item['episode'];assert episode['scene'] in allowed and episode['scene'] not in excluded
  source_path=Path(item['source_rollout']);assert sha256(source_path)==item['source_rollout_sha256'];original=read_json(source_path)
  assert original['episode']==episode and not original['metrics']['success']
  folder.mkdir(exist_ok=True);images=folder/'images';images.mkdir(exist_ok=True)
  rgb=env.reset(episode);history=[];executed=[];frames=[];audit=[];outcomes=[];success=False;verified_prefix_frames=0
  for step in range(32):
   image_hash=pixel_hash(rgb)
   if step<=item['prefix_length'] and step<len(original['steps']):
    assert image_hash==original['steps'][step]['image_sha256'],('Nonidentical original failure replay',item['id'],step);verified_prefix_frames+=1
   image=images/f'{step:03d}.png';Image.fromarray(np.asarray(rgb)[...,:3].astype(np.uint8)).save(image)
   relative=str(image.relative_to(out));history.append(relative);teacher=step>=item['prefix_length']
   if not teacher:action=item['prefix_actions'][step]
   else:
    m=env.private_metrics()
    if m['eligible']:action=0
    else:
     candidates={a:env.probe(a) for a in range(1,17)}
     assert pixel_hash(env.observe()['rgb'])==image_hash,'Counterfactual probe changed the observed state'
     action=env.oracle_action(candidates)
     outcomes.append({'episode_id':item['id'],'source_episode_id':episode['episode_id'],'category':episode['category'],'image':relative,'history':list(history),'previous_actions':list(executed),'observed_collision':[False]+[bool(candidates[a][2]['collision']) for a in range(1,17)],'execution_valid':[True]+[bool(candidates[a][2]['travelled_m']>=.04) for a in range(1,17)]})
   frame={'episode_id':item['id'],'source_episode_id':episode['episode_id'],'category':episode['category'],'image':relative,'image_sha256':sha256(image),'pixel_sha256':image_hash,'history':list(history),'previous_actions':list(executed),'action':int(action),'supervision_mask':teacher}
   response=env.step(action);executed.append(int(action));rgb=response['rgb'];frame['public_feedback']={'collision':bool(response['feedback']['collision'])};frames.append(frame)
   audit.append({'step':step,'action':int(action),'teacher':teacher,'collision':bool(response['feedback']['collision']),'success':bool(response['metrics']['success']),'distance':response['metrics'].get('distance')})
   if not teacher:assert bool(response['feedback']['collision'])==bool(original['steps'][step]['collision'])
   if response['done']:
    assert teacher,'Source prefix unexpectedly ended before recovery';success=bool(response['metrics']['success']);break
  if success:assert frames[-1]['action']==0 and frames[-1]['supervision_mask'] and len(frames)<=32
  atomic_json(folder/'examples.json',frames if success else []);atomic_json(folder/'execution_outcomes.json',outcomes)
  atomic_json(folder/'audit.json',{'item':item,'success':success,'verified_prefix_frames':verified_prefix_frames,'trace':audit,'failed_attempt_rows_used_for_imitation':False})
  row={'id':item['id'],'scene':episode['scene'],'source_episode_id':episode['episode_id'],'success':success,'steps':len(frames),'supervised_frames':sum(r['supervision_mask'] for r in frames) if success else 0,'context_frames':sum(not r['supervision_mask'] for r in frames) if success else 0,'verified_prefix_frames':verified_prefix_frames,'source_rollout_sha256':item['source_rollout_sha256'],'examples_sha256':sha256(folder/'examples.json'),'audit_sha256':sha256(folder/'audit.json'),'outcomes_sha256':sha256(folder/'execution_outcomes.json'),'time':time.time()}
  atomic_json(marker,row);done.append(row);progress(state='collecting',completed=len(done),successful=sum(r['success'] for r in done),last_id=item['id'])
 atomic_json(run/'COMPLETE',{'time':time.time(),'rank':args.rank,'attempts':len(selected),'completed':len(done),'successful':sum(r['success'] for r in done),'records':done});progress(state='complete',completed=len(done),successful=sum(r['success'] for r in done))
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
 if env is not None:env.close()
