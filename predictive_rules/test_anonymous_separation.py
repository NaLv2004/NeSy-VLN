import torch
from predictive_nesy.anonymous_separation import separation
torch.manual_seed(907)
codes=torch.randint(0,64,(2,4,49));logits=torch.randn(2,4,8,requires_grad=True)
p=logits.softmax(-1);valid=torch.ones(2,4,dtype=torch.bool)
loss,metrics=separation(p,codes,valid)
assert metrics['visual_distinct_pairs']==12 and loss>0
renamed,_=separation(p[...,torch.randperm(8)],codes,valid)
assert torch.allclose(loss,renamed,atol=1e-7)
identical=codes[:,:1].expand(-1,4,-1)
zero,m=separation(p,identical,valid);assert zero==0 and m['visual_distinct_pairs']==0
valid[:,-1]=False
base,_=separation(p,codes,valid);changed=p.clone();changed[:,-1]=0
check,_=separation(changed,codes,valid);assert torch.equal(base,check)
unique=torch.eye(8)[:4][None].expand(2,-1,-1)
zero,m=separation(unique,codes);assert zero==0 and m['visual_alias_pair_fraction']==0
loss.backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().max()>0
print(dict(passed=True,renaming_invariant=True,padding_ignored=True,identical_frames_not_negatives=True,finite_nonzero_gradient=True))
