"""Short GPU7 compute-only benchmark using already saved training records."""
import json,os,time,gc
from pathlib import Path
import numpy as np
import torch
from model import FactoredWorld,Config
from closure_objective import ClosureObjective,configure
from accelerated_transition import install
from data import ROOT

assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
torch.set_num_threads(2);torch.manual_seed(631017)
torch.backends.cuda.matmul.allow_tf32=True
DATA=ROOT/'counterfactual_closure_v63';OUT=DATA/'throughput'
saved=torch.load(DATA/'source.pt',map_location='cpu',weights_only=False)
m=FactoredWorld(Config(**saved['model_config'])).cuda();m.load_state_dict(saved['model']);params=configure(m)
keys=('context','memory','observation','first_observation','first_memory','second_actions','second_observation','second_memory','second_valid')
rows={k:[] for k in keys}
for p in sorted((DATA/'episodes').glob('*.npz'))[:32]:
    with np.load(p,allow_pickle=False) as z:
        for k in keys:rows[k].append(z[k].copy())
base={k:torch.from_numpy(np.concatenate(v)).to('cuda',dtype=torch.float32 if k=='context' else torch.bool if k=='second_valid' else torch.long) for k,v in rows.items()}
def batch(n):
    idx=torch.arange(n,device='cuda')%len(base['memory'])
    return {k:v[idx] for k,v in base.items()}
objective=ClosureObjective(m);d=batch(8);pairs=torch.arange(2,10,device='cuda')[None].expand(8,-1)
loss0,metric0=objective(d,pairs);loss0.backward()
g0={k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None}
original=m.transition
install(m);m.zero_grad(set_to_none=True)
loss1,metric1=objective(d,pairs);loss1.backward()
loss_error=abs(float(loss1-loss0));grad_error=max(float((p.grad-g0[k]).abs().max()) for k,p in m.named_parameters() if k in g0)
grad_relative=float(torch.stack([(p.grad-g0[k]).square().sum() for k,p in m.named_parameters() if k in g0]).sum().sqrt()/torch.stack([v.square().sum() for v in g0.values()]).sum().sqrt().clamp_min(1e-12))
assert loss_error<1e-5 and grad_relative<1e-4,(loss_error,grad_error,grad_relative)
assert all(torch.equal(metric0[k],metric1[k]) for k in metric0 if k.endswith(('_hits','_count')))
del g0,loss0,loss1,metric0,metric1,d,pairs
fast=m.transition;results=[]
for name,n in [('legacy',32),('pruned',32),('pruned',128),('pruned',256)]:
    m.transition=original if name=='legacy' else fast
    m.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(params,lr=0.)
    d=batch(n);pairs=torch.arange(2,10,device='cuda')[None].expand(n,-1)
    times=[]
    for i in range(5):
        torch.cuda.synchronize();t=time.perf_counter()
        loss,_=objective(d,pairs);optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step();torch.cuda.synchronize()
        if i:times.append(time.perf_counter()-t)
    row=dict(implementation=name,batch=n,seconds=float(np.median(times)),positions_per_second=n/float(np.median(times)),peak_gib=torch.cuda.max_memory_allocated()/2**30)
    results.append(row);print(json.dumps(row),flush=True)
    del d,pairs,loss,optimizer
record=dict(time=time.time(),loss_difference=loss_error,gradient_max_difference=grad_error,gradient_relative_l2_difference=grad_relative,results=results,
    scope='Existing training records; no validation rollout and no checkpoint changes; optimizer lr0 for throughput only',
    simultaneous_collectors=True)
(OUT/'training_benchmark.json').write_text(json.dumps(record,indent=2))
