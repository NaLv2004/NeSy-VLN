"""Zero residual preserves the source policy; only generic evidence can learn."""
import json,os,time
import torch
from model import FactoredWorld,Config
from data import ROOT,sha

assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
torch.set_num_threads(2)
data=ROOT/'goal_discrimination_v64';out=ROOT/'run_linear_evidence_v65'
source=torch.load(data/'source.pt',map_location='cpu',weights_only=False)
old=FactoredWorld(Config(**source['model_config'])).cuda().eval()
old.load_state_dict(source['model'])
new=FactoredWorld(Config(**dict(source['model_config'],linear_visual_evidence=True))).cuda().eval()
missing,extra=new.load_state_dict(source['model'],strict=False)
assert not extra and set(missing)=={'linear_visual_evidence.weight','evidence_mean','evidence_scale'}
torch.manual_seed(65);x=torch.randn(8,49,128,device='cuda');c=torch.randn(8,128,device='cuda')
with torch.no_grad():
    old_global=old.observe(x);new_global=new.observe(x)
    old_local=old.observe(x,c);new_local=new.observe(x,c)
    assert torch.equal(old_global,new_global) and torch.equal(old_local,new_local)
new.requires_grad_(False);new.linear_visual_evidence.requires_grad_(True)
before={n:p.detach().clone() for n,p in new.named_parameters() if not p.requires_grad}
optimizer=torch.optim.AdamW(new.linear_visual_evidence.parameters(),lr=2e-4)
loss=torch.nn.functional.cross_entropy(new.observe(x,c),torch.arange(8,device='cuda'))
loss.backward();assert new.linear_visual_evidence.weight.grad.abs().sum()>0;optimizer.step()
assert all(torch.equal(before[n],p) for n,p in new.named_parameters() if not p.requires_grad)
assert torch.equal(new.observe(x),old_global),'Global code/context generation changed'
assert not torch.equal(new.observe(x,c),old_local)
result=dict(time=time.time(),zero_adapter_exact_global_and_local=True,original_parameters_unchanged=True,
    updated_global_codes_exactly_unchanged=True,local_all_code_evidence_has_gradient=True,
    runtime_inputs='visual patches and generated definitions; no goal or action',
    disposable_optimizer_steps=1,source_checkpoint_unchanged=True,trainable_parameters=new.linear_visual_evidence.weight.numel())
(out/'contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
