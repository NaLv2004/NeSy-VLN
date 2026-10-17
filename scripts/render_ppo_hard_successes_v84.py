"""Render saved successful PPO actions. No policy inference or new evaluation."""
import os
os.environ['MAGNUM_LOG']='quiet';os.environ['HABITAT_SIM_LOG']='quiet'
import copy,hashlib,json,math,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np,quaternion,habitat_sim
from PIL import Image,ImageDraw,ImageFont

B=Path('/home/user/xjy/nesy_vla_20261010');RUN=B/'parallel_ppo_v84';OUT=B/'hard_success_v84';OUT.mkdir(exist_ok=True)
R=Path('/home/user/xjy/strips_wm_navigation_20261009')
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts'))
sys.path.insert(0,'/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')
from stripsnav.sim_adapter import NavigationEnvironment
from stripsnav.actions import PRIMITIVES
import render_strict_success as overview
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2));q.replace(p)
W,H=768,448;CW,CH=1536,650;FPS=20;SUB=16
overview.WIDTH=W;overview.HEIGHT=H
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fonts={n:ImageFont.truetype(FONT,n) for n in [18,21,25,31]}
cfg=read(R/'configs/experiment.json');cfg['evaluation_horizon']=32;cfg['training_horizon']=32
sys.path.insert(0,cfg['source_sim_code'])
from large_lab_sim import create_sim

def choose():
 candidates=[]
 for folder in sorted((RUN/'rollouts').glob('step_*')):
  update=int(folder.name.split('_')[-1])
  if update<36:continue
  files=sorted(folder.glob('rank*/*.json'))
  if len(files)!=192:continue
  for path in files:
   t=read(path);e=t['episode']
   if t['metrics']['success'] and e['distance_group']=='far' and len(t['steps'])>=10:
    candidates.append((path,t,update))
 candidates.sort(key=lambda x:(x[1]['episode'].get('initial_distance_m',0),len(x[1]['steps'])),reverse=True)
 selected=[];used=set();scenes=set()
 for domain,category in [('dense','chair'),('dense','table'),('normal','chair'),('normal','table'),(None,'chair'),(None,'table')]:
  pool=[x for x in candidates if x[1]['episode']['episode_id'] not in used and
    (domain is None or x[1]['episode']['domain']==domain) and x[1]['episode']['category']==category]
  if not pool:continue
  fresh=[x for x in pool if x[1]['episode']['scene'] not in scenes];x=(fresh or pool)[0]
  selected.append(x);used.add(x[1]['episode']['episode_id']);scenes.add(x[1]['episode']['scene'])
 if len(selected)<4:raise RuntimeError('Not enough completed difficult successes yet; preserve current training and retry selection later.')
 return selected

def pose(state):return {'position':np.asarray(state.position).tolist(),'rotation_wxyz':quaternion.as_float_array(state.rotation).tolist()}
def setpose(sim,pos,q):
 st=habitat_sim.AgentState();st.position=np.asarray(pos,np.float32);st.rotation=q;sim.get_agent(0).set_state(st)
def mark(draw,point,color,r=7):
 x,y=point;draw.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='white',width=2)
def txt(draw,xy,text,size=21,color='#e6eef8'):draw.text(xy,text,fill=color,font=fonts[size])

def render(path,trial,update,index):
 e=trial['episode'];stem=f'{index:02d}_{e["category"]}_{e["domain"]}_{e["episode_id"]}'
 out=OUT/stem;out.mkdir(exist_ok=True)
 if (out/'COMPLETE.json').exists():return read(out/'COMPLETE.json')
 (out/'source_trace.json').write_bytes(path.read_bytes())
 env=NavigationEnvironment(cfg,gpu=0,evaluation=True);poses=[];hash_errors=[]
 try:
  rgb=env.reset(e);initial=env.private_metrics()
  for i,row in enumerate(trial['steps']):
   poses.append(pose(env.state()));actual=hashlib.sha256(np.asarray(rgb)[...,:3].astype(np.uint8).tobytes()).hexdigest()
   assert actual==row['image_sha256'],('Image replay mismatch',e['episode_id'],i)
   response=env.step(row['action'])
   assert response['feedback']['collision']==row['collision']
   for key in ['distance','path_length','visible_pixels']:
    assert abs(response['metrics'][key]-row['metrics'][key])<1e-4,(i,key)
   rgb=response['rgb']
  poses.append(pose(env.state()));assert response['metrics']['success'] and trial['steps'][-1]['action']==0
  camera_cfg=copy.deepcopy(env.cfg)
 finally:env.close()
 write(out/'poses.json',poses)
 scene=Path(cfg['source_world_root'])/e['scene'];cutaway=out/'cutaway.glb';overview.cutaway(scene/'world_mesh.glb',cutaway)
 positions=np.array([p['position'] for p in poses]);points=positions.copy();points[:,1]+=.08
 objects=read(scene/'world.json')['objects'];goals=[o for o in objects if o['name'] in e['goal_object_names']]
 goalpoints=[]
 for o in goals:
  c=np.array(o['center'] if 'center' in o else (np.array(o['aabb_min'])+np.array(o['aabb_max']))/2);c[1]=.12;goalpoints.append(c)
 roi=np.concatenate([points,np.asarray(goalpoints)]);center=(roi.min(0)+roi.max(0))/2;center[1]=.1
 span=max(float(np.ptp(roi[:,[0,2]],axis=0).max())+4,8.)
 over=overview.overview_sim(cutaway,out)
 try:base,project=overview.view(over,center+np.array([.12*span,1.12*span,.62*span]),center)
 finally:over.close()
 route=project(points);targets=project(goalpoints)
 camera_cfg['camera'].update(width=W,height=H);sim=create_sim(camera_cfg,scene,0)
 video=out/'trajectory.mp4'
 process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{CW}x{CH}',
  '-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 n=len(trial['steps']);frames=0;category={'chair':'椅子','table':'桌子'}[e['category']]
 try:
  for i,row in enumerate(trial['steps']):
   p0,p1=positions[i:i+2];q0=np.quaternion(*poses[i]['rotation_wxyz']);q1=np.quaternion(*poses[i+1]['rotation_wxyz'])
   for k in range(SUB if i<n-1 else FPS):
    t=k/(SUB-1) if i<n-1 else 1.
    pos=p0*(1-t)+p1*t;q=quaternion.slerp_evaluate(q0,q1,t);setpose(sim,pos,q)
    rgb=Image.fromarray(sim.get_sensor_observations()['rgb'][...,:3]);top=base.copy();d=ImageDraw.Draw(top)
    walked=[tuple(v) for v in route[:i+1]]+[tuple(project((pos+np.array([0,.08,0]))[None])[0])]
    if len(walked)>1:d.line(walked,fill='#31dfaa',width=6,joint='curve')
    mark(d,route[0],'#399dff',8);mark(d,walked[-1],'#ffffff',8)
    for point in targets:mark(d,point,'#ffc547',10)
    canvas=Image.new('RGB',(CW,CH),'#101c29');canvas.paste(rgb,(0,90));canvas.paste(top,(W,90));d=ImageDraw.Draw(canvas)
    txt(d,(20,8),f'寻找{category}。成功轨迹 {index}。',31)
    difficulty='密集障碍' if e['domain']=='dense' else '普通布局'
    txt(d,(20,51),f'训练采样。起点距目标 {initial["distance"]:.1f} 米。{difficulty}。PPO 更新 {update}。',21)
    txt(d,(20,545),'第一视角',21);txt(d,(W+20,545),'三维俯视。绿色为已执行轨迹，黄色为目标。',21)
    a=row['action'];length,yaw,after=PRIMITIVES[a]
    description='停止，任务成功。' if a==0 else f'{"左转" if yaw>0 else "右转" if yaw<0 else "保持航向"}{str(abs(yaw))+"°" if yaw else ""}，前进 {length:.2f} 米。'
    txt(d,(20,578),f'动作 {i+1}/{n}。{description}',25,color='#45d5b5')
    txt(d,(20,615),f'观测符号 {row["observed_ids"]}。记忆状态 {row["state_ids"]}。',18)
    process.stdin.write(canvas.tobytes());frames+=1
    if i==n//2 and k==SUB//2:canvas.save(out/'preview.jpg',quality=94)
  canvas.save(out/'final.jpg',quality=94)
 finally:
  sim.close();process.stdin.close()
 assert process.wait()==0
 full=base.copy();d=ImageDraw.Draw(full);d.line([tuple(x) for x in route],fill='#31dfaa',width=6,joint='curve')
 mark(d,route[0],'#399dff',8);mark(d,route[-1],'white',8)
 for pt in targets:mark(d,pt,'#ffc547',10)
 full.save(out/'overview.jpg',quality=95)
 manifest={'episode_id':e['episode_id'],'category':e['category'],'domain':e['domain'],'scene':e['scene'],
  'split':'training_rollout','sampling':'stochastic PPO rollout','update':update,'initial_distance_m':initial['distance'],
  'initial_visible_pixels':initial['visible_pixels'],'initially_hidden':initial['visible_pixels']<16,'steps':n,
  'path_length_m':trial['metrics']['path_length'],'final_distance_m':trial['metrics']['distance'],'success':True,
  'collisions':sum(s['collision'] for s in trial['steps']),'video':str(video),'directory':str(out),'frames':frames,'fps':FPS,
  'trace_sha256':sha(path),'image_hashes_replayed_exactly':n,'all_collisions_and_step_metrics_match':True,
  'new_policy_inference':False,'geometry_for_display_only':True,'maps_passed_to_policy':False,'camera_fov_unchanged':True,
  'interpolation':'Rendering between verified saved poses only. No new policy action or navigation evaluation.'}
 write(out/'COMPLETE.json',manifest);return manifest

try:
 selected=choose();write(OUT/'selection.json',[{'path':str(p),'episode_id':t['episode']['episode_id'],'update':u} for p,t,u in selected])
 results=[]
 for i,(path,t,u) in enumerate(selected,1):
  write(OUT/'status.json',{'state':'rendering','pid':os.getpid(),'time':time.time(),'case':i,'completed':len(results)})
  results.append(render(path,t,u,i));write(OUT/'results.json',results)
 write(OUT/'status.json',{'state':'complete','pid':os.getpid(),'time':time.time(),'completed':len(results)})
 print(json.dumps(results,ensure_ascii=False,indent=2))
except Exception as error:
 write(OUT/'status.json',{'state':'failed','pid':os.getpid(),'time':time.time(),'error':repr(error),'traceback':traceback.format_exc()});raise
