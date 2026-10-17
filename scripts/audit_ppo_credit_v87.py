"""Reconstruct actual PPO advantages from an immutable pre-rollout checkpoint.

Uses recorded training integer tokens and actions. No new rollout,
no optimizer update, no development/fixed task data, no simulator.
"""
import fcntl, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import torch

B=Path('/home/user/xjy/nesy_vla_20261010'); run=B/'parallel_ppo_v85'
out=B/'ppo_credit_v87';out.mkdir(exist_ok=True)
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
records=[];episodes=[]
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
write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),time=time.time()))
print(json.dumps(result,indent=2))
