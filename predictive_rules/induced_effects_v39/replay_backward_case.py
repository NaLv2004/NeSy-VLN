"""Exact GPU replay of an actual captured update, no optimizer or simulator."""
import json
import os
from pathlib import Path
import torch
from model import FactoredWorld,hard
from data import ROOT
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
torch.cuda.set_device(0);torch.set_num_threads(3)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
out=ROOT/'run_induced_effects_v39';results=[]
for rank in (0,1):
    case=out/f'backward_case_rank{rank}.pt'
    if not case.exists():
        print(json.dumps(dict(state='awaiting_first_actual_bad_update',rank=rank)));raise SystemExit(3)
    saved=torch.load(case,map_location='cpu',weights_only=False)
    data={k:v.cuda() for k,v in saved['data'].items()}
    for mode in ('full','detach_copy','detach_copy_and_history','detach_copy_and_carrier'):
        model=FactoredWorld().cuda();model.load_state_dict(saved['model'])
        model.detach_copy_evidence=mode!='full';model.detach_memory_carrier=False;model.zero_grad(set_to_none=True)
        if mode=='detach_copy_and_history':
            def sequence(observations,previous,hidden=None):
                carrier=hidden[0] if hidden is not None else None;logits=[];states=[]
                for t in range(observations.shape[1]):
                    lm,state=model.memory_step(carrier,observations[:,t],previous[:,t],initial=carrier is None)
                    logits.append(lm);states.append(state);carrier=hard(lm).detach()
                return torch.stack(logits,1),torch.stack(states,1),carrier[None]
            model.memory_sequence=sequence
        if mode=='detach_copy_and_carrier':
            original=model.memory_step
            def step(memory,observation,previous,initial=False):
                return original(None if memory is None else memory.detach(),observation,previous,initial)
            model.memory_step=step
        torch.set_rng_state(saved['forward_cpu_rng']);torch.cuda.set_rng_state(saved['forward_cuda_rng'])
        loss,metrics=model(data);loss.backward()
        norms={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if p.grad is not None}
        assert float(loss.detach())==saved['loss'],'Backward intervention changed forward loss'
        results.append(dict(rank=rank,update=saved['update'],mode=mode,
            actual_ddp_loss=saved['loss'],replayed_loss=float(loss.detach()),
            actual_ddp_gradient_norm=saved['gradient_norm'],replayed_local_gradient_norm=sum(x*x for x in norms.values())**.5,
            top_parameters=sorted(norms.items(),key=lambda p:-p[1])[:5]))
        del model
    del data,saved
    torch.cuda.empty_cache()
report=dict(results=results,no_optimizer_step=True,no_navigation_rollout=True)
(out/'actual_backward_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
