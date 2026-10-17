"""One resumable held-out evaluation of a saved generated symbolic automaton.

Loads the frozen perceptual frontend identified by the training manifest.
There are no optimizer updates or evaluation-dependent parameter choices.
"""
import argparse,fcntl,gc,hashlib,json,os,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
B=Path('/home/user/xjy/nesy_vla_20261010')
sys.path.insert(0,str(Path(__file__).resolve().parent))
# The experiment snapshot owns the exact automaton implementation. The parent
# directory also contains a legacy nesy_vla package and must not shadow it.
sys.path.insert(0,str(B/'original_ppo_resume_v99/src'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank,image_tensor,pil,load_checkpoint
from nesy_vla.model import NavigationModel
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.evaluation import aggregate


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
 p.add_argument('--suite',choices=['fixed80','development'],default='fixed80')
 p.add_argument('--verify-only',action='store_true');p.add_argument('--verify-replay-dir');a=p.parse_args()
 out=Path(a.out).resolve();assert out.is_relative_to(B);out.mkdir(parents=True,exist_ok=True)
 lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2);torch.manual_seed(5100)
 cfg=read(B/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
 from stripsnav.io import load_committed
 checkpoint=Path(a.checkpoint);commit=read(checkpoint/'COMMITTED');payload=load_committed(checkpoint)
 manifest=payload['manifest'];assert bank.provenance==manifest['data']
 episodes=bank.fixed if a.suite=='fixed80' else bank.dev;assert len(episodes)==80
 episodes=[e for e in episodes if e['domain']=='normal' and e['distance_group'] in ('near','medium')]
 assert episodes and not {e['episode_id'] for e in episodes}.intersection(e['episode_id'] for e in bank.train)
 record={'checkpoint':str(checkpoint),'commit':commit,'epoch':payload['epoch'],'suite':a.suite,
  'episode_ids':[e['episode_id'] for e in episodes],'data_provenance':bank.provenance,
  'horizon':32,'revision_interval':10,'decoding':'greedy','model_training':False,
  'architecture':manifest['architecture'],'frontend_commit':manifest['source_commit'],
  'script_sha256':sha(__file__)}
 if (out/'manifest.json').exists():assert read(out/'manifest.json')==record
 else:atomic(out/'manifest.json',record)
 if (out/'summary.json').exists():
  print(json.dumps({'state':'already_complete','summary':read(out/'summary.json')}));return
 model=ProbabilisticTokenAutomaton(**manifest['architecture']).cuda().eval();model.load_state_dict(payload['model'],strict=True)
 del payload;gc.collect()
 source=manifest['source_commit'];location=Path(source['encoder_source']);vision_cfg=read(location.parent/'config.json')
 assert read(location/'COMMITTED')==source['encoder_source_commit']
 visual=NavigationModel(vision_cfg).eval();load_checkpoint(location,visual);visual.requires_grad_(False).cuda()
 vision=visual.grounder.vision;del visual
 if hasattr(vision,'backbone'):vision=vision.backbone
 assert isinstance(vision,torch.nn.Sequential) and len(vision)==8
 component=Path(source['component_checkpoint']);assert read(component/'COMMITTED')==source['component_commit']
 visual_payload=load_committed(component)
 if visual_payload['visual_tail'] is not None:vision[6:].load_state_dict(visual_payload['visual_tail'])
 del visual_payload;gc.collect()
 assert sha(source['vocabulary_path'])==source['vocabulary_sha256']
 vocabulary=torch.load(source['vocabulary_path'],map_location='cuda',weights_only=True)
 @torch.no_grad()
 def encode(image):
  with torch.autocast('cuda',dtype=torch.bfloat16):f=F.adaptive_avg_pool2d(vision(image_tensor(image,vision_cfg['image_size'],'cuda')),(7,7)).flatten(2).transpose(1,2)
  z=F.normalize(((f.float()-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
  return torch.cdist(z.flatten(0,1),vocabulary['centers']).argmin(1)
 # Check deployment/frontend fidelity only on original training frames.
 encoded=B/'perceptual_sequences_v73'/source['arm'];rows=read(encoded/'training_rows.json');codes=np.load(encoded/'visual_tokens.npy')
 for index in [0,len(rows)//2,len(rows)-1]:
  with Image.open(rows[index]['path']) as image:assert encode(image.convert('RGB')).tolist()==codes[index].tolist()
 if a.verify_only:
  assert a.verify_replay_dir,'Verification requires a completed saved development replay'
  replay=Path(a.verify_replay_dir);assert replay.name==f'step_{record["epoch"]:06d}'
  assert read(replay/'summary.json')['episodes']==80
  selected=bank.dev[:8];traces=[read(replay/e['episode_id']/'trace.json') for e in selected]
  with torch.no_grad():
   grids=[]
   for e,trace in zip(selected,traces):
    with Image.open(replay/e['episode_id']/'first.png') as image:grid=encode(image.convert('RGB'))
    assert grid.tolist()==trace['steps'][0]['patch_tokens'];grids.append(grid)
   x=torch.stack(grids);task=torch.tensor([['chair','table'].index(e['category']) for e in selected],device='cuda')
   _,prior=model.initial_state(8,'cuda');previous_action=torch.full((8,),17,device='cuda',dtype=torch.long)
   program=model.generate(task,x[:,None],0);r=model.step(program,x,prior,previous_action,0)
   proof=model.proof(program,r['observed_ids'],prior,previous_action,r['state_values'],0)
   expected=torch.tensor([trace['steps'][0]['probabilities'] for trace in traces],device='cuda')
   error=float((r['probabilities']-expected).abs().max());assert error==0
   assert torch.equal(proof['probabilities'],r['probabilities'])
   assert r['probabilities'].argmax(-1).tolist()==[trace['steps'][0]['action'] for trace in traces]
  verification={'time':time.time(),'checkpoint':str(checkpoint),'commit':commit,
      'training_frontend_rows_checked':3,'saved_development_first_frames_checked':8,
      'action_probability_replay_max_error':error,'fresh_rollouts':0,'fixed80_observations_used':False,
      'no_optimizer_updates':True,'complete':True}
  atomic(out/'verification.json',verification);print(json.dumps(verification,indent=2));return
 done=[];pending=[]
 for episode in episodes:
  marker=out/episode['episode_id']/'evaluation.json'
  if marker.exists():
   row=read(marker);assert row['checkpoint_sha256']==commit['checkpoint_sha256'] and row['proof_error']==0;done.append(row)
  else:pending.append(episode)
 sims=[];started=time.time()
 try:
  sims=[bank.simulator(out/f'sim_{j}.log',0,evaluation=True) for j in range(8)]
  atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pids':[s.process.pid for s in sims],'time':time.time()})
  with torch.no_grad(),ThreadPoolExecutor(max_workers=8) as pool:
   for offset in range(0,len(pending),8):
    batch=pending[offset:offset+8];n=len(batch)
    responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(sims,batch)))
    images=[pil(r['rgb']) for r in responses];images+=images[:1]*(8-n);task=torch.tensor([['chair','table'].index(e['category']) for e in batch+[batch[0]]*(8-n)],device='cuda')
    memory_ids,memory=model.initial_state(8,'cuda');last_action=torch.full((8,),17,device='cuda',dtype=torch.long)
    active=list(range(n));history=[];trials=[{'episode':e,'steps':[],'programs':[]} for e in batch]
    for e,image in zip(batch,images):
     folder=out/e['episode_id'];folder.mkdir(exist_ok=True);image.save(folder/'first.png')
    for step in range(32):
     x=torch.stack([encode(image) for image in images]);history.append(x)
     if step%10==0:
      program=model.generate(task,torch.stack(history[-11:],1),step)
      for i in active:trials[i]['programs'].append({'start':step,'coefficients':program.coefficients[i].tolist()})
     result=model.step(program,x,memory,last_action,step)
     proof=model.proof(program,result['observed_ids'],memory,last_action,result['state_values'],step)
     assert torch.equal(proof['probabilities'],result['probabilities'])
     chosen=result['probabilities'].argmax(-1)
     for i in active:
      trials[i]['steps'].append({'step':step,'patch_tokens':x[i].tolist(),'observation_tokens':result['observed_ids'][i].tolist(),
       'previous_state':memory_ids[i].tolist(),'previous_action':int(last_action[i]),'state':result['state_ids'][i].tolist(),
       'previous_belief':memory[i].tolist(),'state_belief':result['state_values'][i].tolist(),'action':int(chosen[i]),
       'probabilities':result['probabilities'][i].tolist(),'transition_probabilities':proof['transition_probabilities'][i].tolist(),
       'rule_evidence':proof['rule_evidence'][i].tolist(),'program_start':program.start,
       'image_sha256':hashlib.sha256(np.asarray(images[i]).tobytes()).hexdigest()})
     memory=result['state_values'];memory_ids=result['state_ids'];last_action=chosen
     responses=list(pool.map(lambda pair:sims[pair[0]].call('step',action=pair[1]),[(i,int(chosen[i])) for i in active]))
     remaining=[]
     for i,response in zip(active,responses):
      trials[i]['steps'][-1]['collision']=bool(response.get('feedback',{}).get('collision',False))
      trials[i]['metrics']=response.get('metrics',{});images[i]=pil(response['rgb'])
      if not response['done']:remaining.append(i)
     active=remaining
     if not active:break
    for trial,image in zip(trials,images):
     e=trial['episode'];folder=out/e['episode_id'];image.save(folder/'last.png');atomic(folder/'trace.json',trial,compact=True)
     row={'episode_id':e['episode_id'],'scene':e['scene'],'domain':e['domain'],'category':e['category'],'distance_group':e['distance_group'],
      'success':bool(trial['metrics'].get('success',False)),'steps':len(trial['steps']),'collisions':sum(r['collision'] for r in trial['steps']),
      'checkpoint_sha256':commit['checkpoint_sha256'],'proof_error':0.}
     atomic(folder/'evaluation.json',row);done.append(row)
    progress={'pid':os.getpid(),'state':'evaluating','time':time.time(),'completed':len(done),'total':len(episodes),'summary':aggregate(done)}
    atomic(out/'status.json',progress);print(json.dumps(progress),flush=True)
  summary=dict(aggregate(done),checkpoint=str(checkpoint),checkpoint_sha256=commit['checkpoint_sha256'],epoch=record['epoch'],suite=a.suite,
      seconds=time.time()-started,all_action_rule_proofs_exact=True)
  assert summary['episodes']==len(episodes)
  atomic(out/'summary.json',summary);atomic(out/'status.json',{'pid':os.getpid(),'state':'complete','time':time.time(),'summary':summary})
 except Exception as error:
  atomic(out/'status.json',{'pid':os.getpid(),'state':'failed','time':time.time(),'error':repr(error),'traceback':traceback.format_exc()});raise
 finally:
  for simulator in sims:simulator.close()

if __name__=='__main__':main()
