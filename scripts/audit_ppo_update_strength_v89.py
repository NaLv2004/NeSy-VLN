"""Reconstruct actual PPO advantages from an immutable pre-rollout checkpoint.

Uses recorded training integer tokens and actions. No new rollout,
no optimizer update, no development/fixed task data, no simulator.
"""
import fcntl, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import torch

B=Path('/home/user/xjy/nesy_vla_20261010'); run=B/'parallel_ppo_v85'
out=B/'ppo_update_strength_v89';out.mkdir(exist_ok=True)
sys.path.insert(0,str(run/'src'))
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.automaton_policy_gradient import SymbolicValue,replay,advantages
read=lambda p:json.loads(p.read_text())
def write(path,data):
 p=Path(path);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,p)
lock=(out/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists(),'Audit already completed'
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=True
torch.set_default_device('cuda')
write(out/'process.json',dict(pid=os.getpid(),time=time.time(),scope='Saved training rollout audit, same CUDA math as collection'))
commit=read(run/'step_000060/COMMITTED');path=run/'step_000060'/commit['checkpoint_file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==commit['checkpoint_sha256']
saved=torch.load(path,map_location='cpu',weights_only=False)
assert saved['epoch']==60
model=ProbabilisticTokenAutomaton(**saved['manifest']['architecture']).cuda().eval()
model.load_state_dict(saved['model']);model.requires_grad_(False)
critic=SymbolicValue(model.factors,model.states,model.tasks).cuda().eval()
critic.load_state_dict(saved['critic']);critic.requires_grad_(False)
result=dict(started=time.time(),checkpoint=commit,rollout_update=61,episodes=0,maximum_probability_error=0.,
 maximum_observed_id_mismatch=0,scope='Recorded training rollouts only, no new navigation or optimization')
records=[];episodes=[];recorded_batches=[]
with torch.no_grad():
 for rank in range(8):
  paths=sorted((run/'rollouts/step_000061'/f'rank{rank}').glob('*.json'))
  assert len(paths)==24
  trials=[read(p) for p in paths]
  for start in range(0,24,8):
   batch=trials[start:start+8];length=max(len(t['steps']) for t in batch)
   x=torch.zeros(8,length,49,dtype=torch.long);a=torch.zeros(8,length,dtype=torch.long)
   task=torch.tensor([['chair','table'].index(t['episode']['category']) for t in batch])
   reward=torch.zeros(8,length);valid=torch.zeros(8,length,dtype=torch.bool)
   for i,t in enumerate(batch):
    n=len(t['steps']);valid[i,:n]=True
    x[i]=torch.tensor([s['patch_tokens'] for s in t['steps']]+[t['steps'][-1]['patch_tokens']]*(length-n))
    a[i,:n]=torch.tensor([s['action'] for s in t['steps']]);reward[i,:n]=torch.tensor([s['reward'] for s in t['steps']])
   replayed=replay(model,x,task,a)
   values=critic(replayed['beliefs'].flatten(0,1),task[:,None].expand(8,length).flatten()).reshape(8,length)
   adv,targets=advantages(reward,values,valid)
   recorded_batches.append(dict(codes=x,actions=a,tasks=task,valid=valid,advantages=adv,
     old_probabilities=replayed['probabilities'].detach(),trials=batch))
   mc,_=advantages(reward,torch.zeros_like(values),valid,lam=1.)
   for i,t in enumerate(batch):
    steps=t['steps'];n=len(steps);outcome='success' if t['metrics']['success'] else ('false_stop' if steps[-1]['action']==0 else 'horizon')
    expected=torch.tensor([s['probabilities'] for s in steps])
    error=float((expected-replayed['probabilities'][i,:n]).abs().max())
    result['maximum_probability_error']=max(error,result['maximum_probability_error'])
    mismatch=int((replayed['observed_ids'][i,:n]!=torch.tensor([s['observed_ids'] for s in steps])).sum())
    result['maximum_observed_id_mismatch']=max(mismatch,result['maximum_observed_id_mismatch'])
    assert error<2e-5 and mismatch==0,dict(rank=rank,trajectory=i+start,error=error,mismatch=mismatch)
    episodes.append(dict(outcome=outcome,return_sum=float(reward[i,:n].sum()),length=n,episode_id=t['episode']['episode_id']))
    for j,s in enumerate(steps):
     p=replayed['probabilities'][i,j]
     records.append(dict(outcome=outcome,action=s['action'],collision=s['collision'],
       repeated_collision=j>0 and steps[j-1]['collision'] and s['action']==steps[j-1]['action'],
       repeated_any='repeat_action' in s['reward_terms'],reward=float(reward[i,j]),value=float(values[i,j]),
       gae=float(adv[i,j]),target=float(targets[i,j]),mc_return=float(mc[i,j]),
       p_stop=float(p[0]),p_chosen=float(p[s['action']]),p_max=float(p.max()),entropy=float(-(p*p.clamp_min(1e-20).log()).sum())))
   result['episodes']+=8
  write(out/'status.json',dict(state='auditing',pid=os.getpid(),rank_completed=rank,time=time.time()))
assert result['episodes']==192
raw=torch.tensor([r['gae'] for r in records],dtype=torch.float64);mean=raw.mean();std=raw.std(correction=0)
for r in records:r['normalized_advantage']=(r['gae']-float(mean))/float(std)
groups=defaultdict(list)
for r in records:
 for label in ['all','outcome/'+r['outcome']]:groups[label].append(r)
 for flag in ['collision','repeated_collision','repeated_any']:
  if r[flag]:groups[flag].append(r)
 if r['action']==0:groups['STOP/'+r['outcome']].append(r)
def summarize(rs):
 fields=['reward','value','gae','normalized_advantage','target','mc_return','p_stop','p_chosen','p_max','entropy']
 v=torch.tensor([r['value'] for r in rs]);m=torch.tensor([r['mc_return'] for r in rs])
 return dict(count=len(rs),**{k:sum(r[k] for r in rs)/len(rs) for k in fields},
  positive_advantage_fraction=sum(r['normalized_advantage']>0 for r in rs)/len(rs),
  value_mc_explained_variance=float(1-(m-v).var(correction=0)/m.var(correction=0)) if float(m.var(correction=0))>1e-10 else None)
result.update(finished=time.time(),actions=len(records),advantage_mean=float(mean),advantage_std=float(std),
 groups={k:summarize(v) for k,v in groups.items()},episode_outcomes={k:sum(e['outcome']==k for e in episodes) for k in ['success','false_stop','horizon']},
 episode_return_means={k:sum(e['return_sum'] for e in episodes if e['outcome']==k)/max(1,sum(e['outcome']==k for e in episodes)) for k in ['success','false_stop','horizon']},
 qualification='Normalized-advantage signs describe the actual surrogate direction for recorded actions, before shared-parameter coupling. Critic explained variance against sampled Monte Carlo returns is noisy and does not alone establish causality.')


# This is an optimizer preflight on recorded ORIGINAL TRAINING tasks. It does
# not replace the main training model and cannot establish navigation gains.
import ast,copy,random,numpy as np
import torch.nn.functional as F
from nesy_vla.automaton_policy_gradient import policy_objective
from nesy_vla.observation_inverse import observation_inverse_logits
source=B/'perceptual_sequences_v73/tail_1e4'
metadata=read(source/'TOKENS_COMMITTED');rows=read(source/'training_rows.json')
assert hashlib.sha256((source/'training_rows.json').read_bytes()).hexdigest()==metadata['training_rows_sha256']
assert metadata==saved['manifest']['source_commit']
codes=np.load(source/'visual_tokens.npy');all_codes=torch.tensor(codes.astype(np.int64),device='cuda')
labels=torch.tensor([r['label'] for r in rows],device='cuda');supervised=torch.tensor([r['supervised'] for r in rows],device='cuda')
weight=torch.tensor([4. if i>=metadata['old_count'] else 1. for i in range(len(rows))],device='cuda')
demo_groups={}
for i,row in enumerate(rows):demo_groups.setdefault(row['episode_id'],[]).append(i)
demos=[dict(id=eid,task=['chair','table'].index(rows[indices[0]]['category']),indices=indices) for eid,indices in demo_groups.items()]
tree=ast.parse((run/'src/train.py').read_text())
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='demonstration_loss')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(run/'src/train.py'),'exec'),globals())
model.requires_grad_(True);model.train()
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.0001)
optimizer.load_state_dict(saved['optimizer'])
for group in optimizer.param_groups:group['lr']=3e-4
rng=random.Random(8900);total_actions=sum(int(b['valid'].sum()) for b in recorded_batches)
result['preflight']=dict(actor_lr=3e-4,max_epochs=8,anchor_weight=.1,mean_kl_limit=.02,epochs=[],
 qualification='One candidate only. Eight-rank actor gradients emulated by serial accumulation of all192recordedrollouts plus64SFTepisodes. No new navigation, no validation fitting, no promoted checkpoint.')
for epoch in range(8):
 started=time.time();snapshot=copy.deepcopy(model.state_dict());optimizer_snapshot=copy.deepcopy(optimizer.state_dict())
 optimizer.zero_grad(set_to_none=True);surrogate=0.;anchor_value=0.
 for b in recorded_batches:
  r=replay(model,b['codes'],b['tasks'],b['actions']);mask=b['valid']
  oldlog=b['old_probabilities'].clamp_min(1e-20).log().gather(-1,b['actions'][...,None]).squeeze(-1)
  loss,audit=policy_objective(r['probabilities'],b['actions'],oldlog,(b['advantages']-float(mean))/float(std),mask)
  weighted=loss*int(mask.sum())/total_actions;weighted.backward();surrogate+=float(weighted.detach())
  del r,loss,weighted
 for _ in range(8):
  anchor=demonstration_loss([demos[rng.randrange(len(demos))] for _ in range(8)])
  (.1*anchor/8).backward();anchor_value+=float(anchor.detach())/8;del anchor
 norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm)
 optimizer.step()
 kl_sum=0.;changed=0;clipped=0;deltas=defaultdict(list);new_objective=0.
 with torch.no_grad():
  for b in recorded_batches:
   r=replay(model,b['codes'],b['tasks'],b['actions']);p=r['probabilities'];old=b['old_probabilities'];mask=b['valid']
   oldlog=old.clamp_min(1e-20).log();newlog=p.clamp_min(1e-20).log()
   kl_sum+=float((old*(oldlog-newlog)).sum(-1)[mask].sum())
   changed+=int(((old.argmax(-1)!=p.argmax(-1))&mask).sum())
   taken_old=oldlog.gather(-1,b['actions'][...,None]).squeeze(-1);taken_new=newlog.gather(-1,b['actions'][...,None]).squeeze(-1)
   clipped+=int(((taken_new-taken_old).exp()[mask]-1).abs().gt(.2).sum())
   loss,_=policy_objective(p,b['actions'],taken_old,(b['advantages']-float(mean))/float(std),mask)
   new_objective+=float(loss)*int(mask.sum())/total_actions
   for i,t in enumerate(b['trials']):
    for j,s in enumerate(t['steps']):
     label='success_stop' if s['action']==0 and t['metrics']['success'] else 'false_stop' if s['action']==0 else None
     if label:deltas[label].append(float(taken_new[i,j]-taken_old[i,j]))
     if j and t['steps'][j-1]['collision'] and s['action']==t['steps'][j-1]['action']:
      deltas['repeat_after_collision'].append(float(taken_new[i,j]-taken_old[i,j]))
 row=dict(epoch=epoch+1,seconds=time.time()-started,pre_update_surrogate=surrogate,post_update_surrogate=new_objective,
  anchor_loss=anchor_value,gradient_norm=norm,mean_kl=max(0.,kl_sum/total_actions),changed_greedy_fraction=changed/total_actions,
  clip_fraction=clipped/total_actions,action_log_probability_change={k:dict(count=len(v),mean=sum(v)/len(v),
    fraction_decreased=sum(x<0 for x in v)/len(v)) for k,v in deltas.items()})
 row['accepted']=np.isfinite(row['mean_kl']) and row['mean_kl']<=.02
 if not row['accepted']:
  model.load_state_dict(snapshot);optimizer.load_state_dict(optimizer_snapshot)
 result['preflight']['epochs'].append(row)
 write(out/'status.json',dict(state='optimizer_preflight',pid=os.getpid(),time=time.time(),latest=row));print(json.dumps(row),flush=True)
 if not row['accepted']:break
result['finished']=time.time();result['preflight']['accepted_epochs']=sum(r['accepted'] for r in result['preflight']['epochs'])
write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),time=time.time()))
print(json.dumps(result['preflight'],indent=2))
