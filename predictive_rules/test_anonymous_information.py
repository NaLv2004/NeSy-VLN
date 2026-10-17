import torch
from predictive_nesy.anonymous_information import separation
torch.manual_seed(906)
codes=torch.randint(0,64,(2,4,49));p=torch.eye(8)[:4][None].expand(2,-1,-1)
identified,m=separation(p,codes);uniform,_=separation(torch.full_like(p,1/8),codes)
assert torch.allclose(identified,torch.tensor(-1.)) and abs(float(uniform))<1e-6
assert identified<uniform and m['visual_alias_pair_fraction']==0
renamed,_=separation(p[...,torch.randperm(8)],codes);assert torch.equal(identified,renamed)
same,m=separation(p,codes[:,:1].expand(-1,4,-1));assert same==0 and m['visual_distinct_pairs']==0
duplicated,_=separation(torch.cat([p,p[:,:1]],1),torch.cat([codes,codes[:,:1]],1));assert torch.allclose(identified,duplicated)
logits=torch.randn(2,4,8,requires_grad=True);loss,_=separation(logits.softmax(-1),codes);loss.backward()
assert torch.isfinite(logits.grad).all() and logits.grad.abs().max()>0
print(dict(passed=True,uniform_flattening_not_a_solution=True,renaming_invariant=True,duplicate_weight_invariant=True,finite_nonzero_gradient=True))
