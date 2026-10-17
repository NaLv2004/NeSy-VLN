"""Find and replay one bad backward pass; no optimizer/environment changes."""
import json
import random
import os
import torch
from data import Corpus,ROOT
from model import FactoredWorld

torch.set_num_threads(3)
device='cuda' if os.environ.get('AUDIT_USE_CUDA')=='1' else 'cpu'
if device=='cuda':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
out=ROOT/'run_induced_effects_v39'
commit=json.loads((out/'latest.COMMITTED.json').read_text())
payload=torch.load(out/commit['filename'],map_location='cpu',weights_only=False)
corpus=Corpus(1,2);rng=random.Random(39104)
model=FactoredWorld().to(device);model.load_state_dict(payload['model'])
def backward(data,detach):
    model.detach_copy_evidence=detach;model.detach_memory_carrier=False;model.zero_grad(set_to_none=True);torch.manual_seed(39103)
    loss,metrics=model(data);loss.backward()
    norms={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if p.grad is not None}
    return dict(loss=float(loss.detach()),gradient_norm=sum(x*x for x in norms.values())**.5,
        top_parameters=sorted(norms.items(),key=lambda p:-p[1])[:5])
tested=[];result=None
for attempt in range(12):
    data=corpus.sample(rng,128,64,device)
    before=backward(data,False);tested.append(before['gradient_norm'])
    if before['gradient_norm']>1000:
        after=backward(data,True)
        assert before['loss']==after['loss'],'Detached copy changed forward computation'
        result=dict(attempt=attempt+1,before=before,after=after,
            reduction_factor=before['gradient_norm']/after['gradient_norm']);break
report=dict(source_commit=commit,tested_gradient_norms=tested,result=result,device=device,branch_rank=1,
    no_optimizer_steps=True,no_rollouts=True)
(out/f'copy_gradient_audit_{device}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
