"""Post-hoc simulator measurements of archived symbolic states, never policy inputs."""
import os
os.environ['MAGNUM_LOG']='quiet';os.environ['HABITAT_SIM_LOG']='quiet'
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,math,sys,time,traceback
from pathlib import Path
import numpy as np,quaternion
from PIL import Image
B=Path('/home/user/xjy/nesy_vla_20261010');RUN=B/'parallel_ppo_v85'
OUT=B/'state_semantics_v91';OUT.mkdir(exist_ok=True)
R=Path('/home/user/xjy/strips_wm_navigation_20261009');sys.path.insert(0,str(R))
from stripsnav.sim_adapter import NavigationEnvironment
from stripsnav.actions import PRIMITIVES
read=lambda p:json.loads(p.read_text())
def write(name,data):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(data,ensure_ascii=False));q.replace(p)
source=RUN/'evaluations/step_000070';paths=sorted(source.glob('*/trace.json'))
assert len(paths)==80
cfg=read(R/'configs/experiment.json');cfg.update(evaluation_horizon=32,training_horizon=32)
env=NavigationEnvironment(cfg,gpu=0,evaluation=True)
rows=[];checks=[];started=time.time();(OUT/'frames').mkdir(exist_ok=True)
write('process.json',dict(pid=os.getpid(),started=started,source=str(source)))
try:
 for j,path in enumerate(paths):
  t=read(path);e=t['episode'];rgb=env.reset(e)
  objects=read(Path(cfg['source_world_root'])/e['scene']/'world.json')['objects']
  targets=[o for o in objects if o['name'] in e['goal_object_names']]
  centers=np.asarray([o.get('center',(np.asarray(o['aabb_min'])+np.asarray(o['aabb_max']))/2) if 'aabb_min' in o else o['center'] for o in targets])
  assert len(centers)
  for i,s in enumerate(t['steps']):
   assert hashlib.sha256(np.asarray(rgb)[...,:3].astype(np.uint8).tobytes()).hexdigest()==s['image_sha256'],(e['episode_id'],i,'RGB mismatch')
   obs=env.observe();state=env.state();metrics=env.private_metrics(obs)
   cam=state.sensor_states['depth_gt'];h,w=obs['depth_gt'].shape
   # Nearest target center bearing in the robot frame, including invisible targets.
   delta=centers-np.asarray(state.position);idx=np.linalg.norm(delta[:,[0,2]],axis=1).argmin()
   local=quaternion.rotate_vectors(state.rotation.conjugate(),delta[idx])
   bearing=math.degrees(math.atan2(-float(local[0]),-float(local[2]))) # positive left
   semantic=np.isin(obs['semantic'],env.env.target_render_ids);ys,xs=np.where(semantic)
   image_side=('not_visible' if len(xs)<16 else 'left' if xs.mean()/w<.4 else 'right' if xs.mean()/w>.6 else 'center')
   # Sparse RGB-D surface measurements, filtered to physical robot height.
   yy,xx=np.mgrid[0:h:4,0:w:4];z=obs['depth_gt'][::4,::4]
   f=w/(2*math.tan(math.radians(env.cfg['camera']['hfov_deg'])/2))
   pc=np.stack([(xx-(w-1)/2)*z/f,-(yy-(h-1)/2)*z/f,-z],-1)
   world=quaternion.rotate_vectors(cam.rotation,pc)+np.asarray(cam.position)
   robot=quaternion.rotate_vectors(state.rotation.conjugate(),world-np.asarray(state.position))
   angles=np.degrees(np.arctan2(-robot[...,0],-robot[...,2]));radii=np.linalg.norm(robot[...,[0,2]],axis=-1)
   valid=(z>.02)&np.isfinite(z)&(world[...,1]-state.position[1]>.08)&(world[...,1]-state.position[1]<.60)
   surfaces={}
   for name,lo,hi in [('left',15,65),('center',-15,15),('right',-65,-15)]:
    mask=valid&(angles>=lo)&(angles<hi)
    surfaces[name]=float(np.quantile(radii[mask],.1)) if mask.sum()>=3 else None
   # Independent collision probes in three directions. Restore pose after each.
   probes={}
   for name,a in [('left',6),('center',4),('right',2)]:
    _,_,m=env.probe(a);probes[name]=dict(collision=m['collision'],travelled_m=m['travelled_m'])
   assert np.allclose(env.state().position,state.position)
   frame=f'frames/{e["episode_id"]}_{i:02d}.jpg';Image.fromarray(np.asarray(rgb)[...,:3]).save(OUT/frame,quality=90)
   length,yaw,after=PRIMITIVES[s['action']]
   rows.append(dict(episode_id=e['episode_id'],scene=e['scene'],category=e['category'],domain=e['domain'],
    initial_distance=e['initial_distance_m'],step=i,program_start=s['program_start'],
    observed=s['observation_tokens'],memory=s['state'],belief=s['state_belief'],
    previous_belief=s['previous_belief'],previous_action=s['previous_action'],
    action=s['action'],action_group='stop' if s['action']==0 else 'left' if yaw>0 else 'right' if yaw<0 else 'forward',
    action_yaw=yaw,action_length=length,probabilities=s['probabilities'],collision=s['collision'],
    outcome_success=t['metrics']['success'],distance=metrics['distance'],eligible=metrics['eligible'],
    target_visible=len(xs)>=16,target_pixels=len(xs),target_image_x=float(xs.mean()/w) if len(xs) else None,
    target_image_side=image_side,target_bearing=bearing,
    target_bearing_bin='behind' if abs(bearing)>90 else 'left' if bearing>20 else 'right' if bearing< -20 else 'front',
    distance_bin='near' if metrics['distance']<=1 else 'middle' if metrics['distance']<=5 else 'far',
    obstacles=probes,surface_distance=surfaces,frame=frame,image_sha256=s['image_sha256']))
   response=env.step(s['action']);assert bool(response['feedback']['collision'])==bool(s['collision'])
   rgb=response['rgb']
  for k in ['distance','path_length','visible_pixels','success']:
   assert abs(float(response['metrics'][k])-float(t['metrics'][k]))<1e-4,(e['episode_id'],k)
  checks.append(dict(episode_id=e['episode_id'],steps=len(t['steps']),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  write('status.json',dict(state='replaying',episodes=j+1,steps=len(rows),time=time.time(),pid=os.getpid()))
 write('measurements.json',dict(source=str(source),checkpoint=read(RUN/'best_development/COMMITTED'),
   rows=rows,checks=checks,started=started,finished=time.time(),
   scope='Development trajectories already evaluated at PPO70. Exact action replay only; no fresh policy inference, no training labels or policy maps. Obstacles measured with depth and independent restored-pose collision probes.'))
 write('status.json',dict(state='complete',episodes=len(checks),steps=len(rows),time=time.time(),pid=os.getpid()))
except Exception:
 write('status.json',dict(state='failed',time=time.time(),pid=os.getpid(),traceback=traceback.format_exc()));raise
finally:env.close()
