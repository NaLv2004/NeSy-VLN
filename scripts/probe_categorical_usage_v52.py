"""Short training-only control: soft-score balance versus actual symbol usage.

This is a quantizer-learning diagnostic, not a navigation benchmark. No result
or checkpoint from it is automatically promoted into navigation training.
"""
import gc,json,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
b=Path('/home/user/xjy/nesy_vla_20261010');sys.path.insert(0,str(b/'symbolic_paired_v12'))
from nesy_vla.runtime import read,sha,atomic
from nesy_vla.factorized_automaton import FactorizedTokenAutomaton,categorical_usage_loss
torch.set_num_threads(2);assert torch.cuda.mem_get_info()[0]>16*2**30
out=b/'categorical_usage_probe_v52';out.mkdir(exist_ok=True)
if (out/'result.json').exists():raise SystemExit('Probe already complete; do not repeat')
source=b/'spatial_automaton_v43';rows=read(source/'training_rows.json');codes=np.load(source/'visual_tokens.npy')
all_indices=np.array([i for i,r in enumerate(rows) if r['supervised']]);rng=np.random.default_rng(5200);rng.shuffle(all_indices)
fit_indices=all_indices[:8192];audit_indices=all_indices[8192:10240]
labels=torch.tensor([r['label'] for r in rows],device='cuda');tasks=torch.tensor([['chair','table'].index(r['category']) for r in rows],device='cuda')
x=torch.from_numpy(codes.astype(np.int64)).cuda();schedule=[rng.choice(fit_indices,128,replace=False) for _ in range(200)]
result={'time':time.time(),'source_rows_sha256':sha(source/'training_rows.json'),'source_tokens_sha256':sha(source/'visual_tokens.npy'),
 'fit_indices':fit_indices.tolist(),'audit_indices':audit_indices.tolist(),'weight':.05,'updates':200,'arms':{},
 'qualification':'Single-frame predicate-learning diagnostic using only original training rows. Audit rows are excluded from this probe fitting. No navigation success or fixed80 evaluation.'}
for name,hard in [('soft_scores',False),('hard_assignments',True)]:
 torch.manual_seed(5100);m=FactorizedTokenAutomaton().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);measurements=[];started=time.time()
 for step,index in enumerate(schedule,1):
  ids=torch.tensor(index,device='cuda');program=m.generate(tasks[ids],x[ids,None],0);_,prior=m.initial_state(len(ids),'cuda');last=torch.full_like(ids,17)
  r=m.step(program,x[ids],prior,last,0)
  action=F.nll_loss(r['probabilities'].clamp_min(1e-20).log(),labels[ids]);recon=F.cross_entropy(r['reconstruction'].transpose(1,2),x[ids])
  balance=sum(categorical_usage_loss(p,torch.ones(len(ids),device='cuda',dtype=torch.bool),hard) for p in [r['observed_probabilities'],r['state_probabilities']])
  loss=action+.2*recon+.05*balance;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
  if step in [25,50,100,200]:
   with torch.no_grad():
    counts=torch.zeros(4,16,device='cuda');correct=total=0;reconstruction=0.;soft=hard_marginal=0.
    for start in range(0,len(audit_indices),128):
     ids=torch.tensor(audit_indices[start:start+128],device='cuda');p=m.generate(tasks[ids],x[ids,None],0);_,v=m.initial_state(len(ids),'cuda');r=m.step(p,x[ids],v,torch.full_like(ids,17),0)
     counts+=r['observed_values'].sum(0);correct+=int((r['probabilities'].argmax(-1)==labels[ids]).sum());total+=len(ids)
     reconstruction+=float(F.cross_entropy(r['reconstruction'].transpose(1,2),x[ids]))*len(ids)
     valid=torch.ones(len(ids),device='cuda',dtype=torch.bool)
     soft+=float(categorical_usage_loss(r['observed_probabilities'],valid))*len(ids)
     hard_marginal+=float(categorical_usage_loss(r['observed_probabilities'],valid,True))*len(ids)
    measurements.append({'step':step,'audit_action_accuracy':correct/total,'audit_reconstruction_ce':reconstruction/total,
      'observation_ids_used':(counts>0).sum(-1).tolist(),'soft_balance_loss':soft/total,'hard_balance_loss':hard_marginal/total,
      'largest_observation_id_fraction_per_factor':(counts/counts.sum(-1,keepdim=True)).max(-1).values.tolist()})
 result['arms'][name]={'seconds':time.time()-started,'measurements':measurements};atomic(out/'progress.json',result)
 del m,opt;gc.collect();torch.cuda.empty_cache()
result['finished']=time.time();atomic(out/'result.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['fit_indices','audit_indices']},indent=2))
