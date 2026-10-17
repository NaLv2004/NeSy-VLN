"""Matched gradient audit; stop-gradient changes no categorical forward value.

Only original training demonstrations are used. No simulator or fixed80 access.
"""
import copy,fcntl,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'temporal_symbol_gradient_v58';out.mkdir(exist_ok=True)
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));sys.path.insert(1,str(b/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic,setup,DataBank
from nesy_vla.factorized_automaton import FactorizedTokenAutomaton
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists(),'Completed diagnostic must not be repeated'
torch.set_num_threads(2);torch.manual_seed(5800)
cfg=read(b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
from stripsnav.io import load_committed
source=b/'spatial_automaton_v43';extra=b/'data/onpolicy_recovery_v48'
rows=read(source/'training_rows.json');old_count=len(rows);rows+=read(extra/'training_rows.json')
assert bank.provenance==read(source/'COMMITTED')['data_provenance']
assert sha(extra/'visual_tokens.npy')==read(extra/'TOKENS_COMMITTED')['tokens_sha256']
codes=np.concatenate([np.load(source/'visual_tokens.npy'),np.load(extra/'visual_tokens.npy')])
groups={}
for i,row in enumerate(rows):groups.setdefault(row['episode_id'],[]).append(i)
assert not set(groups).intersection(e['episode_id'] for e in bank.dev+bank.fixed)
episodes=[dict(id=eid,indices=indices,task=['chair','table'].index(rows[indices[0]]['category'])) for eid,indices in groups.items()]
rng=random.Random(5800);rng.shuffle(episodes)
batches=[episodes[start:start+64] for start in range(0,1024,64)]
all_codes=torch.tensor(codes.astype(np.int64),device='cuda')
labels=torch.tensor([r['label'] for r in rows],device='cuda')
supervised=torch.tensor([r['supervised'] for r in rows],device='cuda')
weight=torch.tensor([4. if i>=old_count else 1. for i in range(len(rows))],device='cuda')
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)

def compute(model,batch,detach):
 n=len(batch);length=max(len(e['indices']) for e in batch)
 indices=torch.zeros(n,length,dtype=torch.long,device='cuda');valid=torch.zeros_like(indices,dtype=torch.bool)
 for j,e in enumerate(batch):
  count=len(e['indices']);indices[j,:count]=torch.tensor(e['indices'],device='cuda');valid[j,:count]=True
 x=all_codes[indices];y=labels[indices];weights=weight[indices]*valid*supervised[indices]
 task=torch.tensor([e['task'] for e in batch],device='cuda')
 _,previous=model.initial_state(n,'cuda');last_action=torch.full((n,),17,device='cuda',dtype=torch.long)
 before_state=None;total=torch.tensor(0.,device='cuda');predictions=[]
 denominator=valid.sum().clamp_min(1)
 for step in range(length):
  if step%10==0:program=model.generate(task,x[:,max(0,step-10):step+1],step)
  r=model.step(program,x[:,step],previous.detach() if detach else previous,last_action,step)
  predictions.append(r['probabilities'].detach())
  active=valid[:,step].float()
  action=-r['probabilities'].clamp_min(1e-20).log().gather(1,y[:,step,None]).squeeze(1)
  loss=(action*weights[:,step]).sum()/weights.sum().clamp_min(1)
  reconstruct=F.cross_entropy(r['reconstruction'].transpose(1,2),x[:,step],reduction='none').mean(-1)
  loss=loss+.2*(reconstruct*active).sum()/denominator
  if step+1<length:
   prediction=model.next_observation(r['state_values'],y[:,step])
   pred=F.cross_entropy(prediction.transpose(1,2),x[:,step+1],reduction='none').mean(-1)
   loss=loss+.2*(pred*valid[:,step+1]).sum()/denominator
  if before_state is not None:
   inverse=model.inverse_action(before_state,r['state_values'])
   loss=loss+.1*(F.cross_entropy(inverse,y[:,step-1],reduction='none')*active).sum()/denominator
  for distribution in [r['observed_probabilities'],r['state_probabilities']]:
   mean=(distribution*active[:,None,None]).sum(0)/active.sum().clamp_min(1)
   loss=loss+.005*(mean*(mean.clamp_min(1e-12)*model.states).log()).sum(-1).mean()/length
  total=total+loss;previous=r['state_values'];before_state=previous;last_action=y[:,step]
 model.zero_grad(set_to_none=True);total.backward()
 by_module={};total_squared=0.
 for name,param in model.named_parameters():
  if param.grad is None:continue
  squared=float(param.grad.detach().float().square().sum());key=name.split('.')[0]
  by_module[key]=by_module.get(key,0.)+squared;total_squared+=squared
 norm=total_squared**.5
 return {'loss':float(total.detach()),'gradient_norm':norm,'global_clip5_scale':min(1.,5/max(norm,1e-12)),
 'group_gradient_norms':{k:v**.5 for k,v in by_module.items()}},torch.stack(predictions)

result={'time':time.time(),'batches':[[e['id'] for e in batch] for batch in batches],
 'data':bank.provenance,'official_heldout_used':False,'optimizer_updates':0,'arms':{}}
try:
 for arm in ['bridge_only','goal_programs']:
  loc=b/'goal_conditioned_automaton_v55'/arm/'step_000040';saved=load_committed(loc)
  model=FactorizedTokenAutomaton(**saved['manifest']['architecture']).cuda().train();model.load_state_dict(saved['model'])
  measures=[]
  for i,batch in enumerate(batches):
   full,p=compute(model,batch,False);cut,q=compute(model,batch,True)
   error=float((p-q).abs().max());assert error==0 and full['loss']==cut['loss']
   measures.append({'batch':i,'full_bptt':full,'detached_previous_state':cut,'forward_probability_error':error})
   progress(state='computing',arm=arm,batch=i+1,total=len(batches))
  norms={mode:np.array([r[mode]['gradient_norm'] for r in measures]) for mode in ['full_bptt','detached_previous_state']}
  result['arms'][arm]={'source_commit':read(loc/'COMMITTED'),'measurements':measures,
   'summary':{mode:{'median_norm':float(np.median(x)),'max_norm':float(x.max()),'batches_above100':int((x>100).sum()),'batches_clipped':int((x>5).sum())} for mode,x in norms.items()}}
  del model,saved;torch.cuda.empty_cache()
 result['finished']=time.time();atomic(out/'result.json',result);progress(state='complete',summary={k:v['summary'] for k,v in result['arms'].items()})
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
