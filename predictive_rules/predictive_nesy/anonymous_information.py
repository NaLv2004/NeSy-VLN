"""Preserve information in hard anonymous state assignments.

Minimize conditional entropy minus marginal entropy within an observed window.
Uniformly flattening every state distribution gives zero information, rather
than the low-overlap solution exploited by the earlier repulsion loss.
No task outcomes or manually meaningful state identities are supplied.
"""
import torch


def separation(posterior,codes,valid=None):
    if posterior.ndim!=3 or codes.shape[:2]!=posterior.shape[:2]:raise ValueError('Expected batched sequences')
    if valid is None:valid=torch.ones(codes.shape[:2],device=codes.device,dtype=torch.bool)
    valid=valid.bool();same=codes[:,:,None].eq(codes[:,None]).all(-1)
    multiplicity=(same & valid[:,None]).sum(-1).clamp_min(1)
    weights=valid.float()/multiplicity;unique=weights.sum(-1);weights=weights/unique[:,None].clamp_min(1)
    with torch.autocast(device_type=posterior.device.type,enabled=False):
        p=posterior.float();entropy=-(p*p.clamp_min(1e-8).log()).sum(-1)
        conditional=(entropy*weights).sum(-1);marginal=(p*weights[:,:,None]).sum(1)
        marginal_entropy=-(marginal*marginal.clamp_min(1e-8).log()).sum(-1)
        usable=unique>1.001
        loss=(((conditional-marginal_entropy)/unique.clamp_min(2).log())*usable).sum()/usable.sum().clamp_min(1)
    mask=(~same)&valid[:,:,None]&valid[:,None]
    mask=mask & torch.ones(mask.shape[-2:],device=mask.device,dtype=torch.bool).triu(1)
    ids=p.argmax(-1);alias=(((ids[:,:,None]==ids[:,None])&mask).float().sum()/mask.sum().clamp_min(1))
    return loss,dict(visual_alias_pair_fraction=float(alias.detach()),visual_distinct_pairs=int(mask.sum()),
        state_conditional_entropy=float(conditional.mean().detach()),state_marginal_entropy=float(marginal_entropy.mean().detach()),
        state_mutual_information=float((marginal_entropy-conditional).mean().detach()))
