"""Visual instance discrimination for anonymous categorical states.

No coordinate, distance, outcome, collision label or semantic state name enters
this loss. It is training-only: distinguish observed inputs that actually differ.
Identical inputs and padding are not negative examples.
"""
import torch


def separation(posterior,codes,valid=None):
    if posterior.ndim!=3 or codes.shape[:2]!=posterior.shape[:2]:raise ValueError('Expected batched sequences')
    if valid is None:valid=torch.ones(codes.shape[:2],device=codes.device,dtype=torch.bool)
    different=codes[:,:,None].ne(codes[:,None]).any(-1)
    mask=different & valid[:,:,None].bool() & valid[:,None].bool()
    mask=mask & torch.ones(mask.shape[-2:],device=mask.device,dtype=torch.bool).triu(1)
    # Keep probabilities in FP32 so near-one overlaps are not rounded to one.
    with torch.autocast(device_type=posterior.device.type,enabled=False):
        overlap=torch.einsum('bts,bus->btu',posterior.float(),posterior.float())
        loss=-(torch.log1p(-overlap.clamp(max=1-1e-6))*mask).sum()/mask.sum().clamp_min(1)
    ids=posterior.argmax(-1)
    aliases=(((ids[:,:,None]==ids[:,None])&mask).float().sum()/mask.sum().clamp_min(1))
    return loss,dict(visual_alias_pair_fraction=float(aliases.detach()),visual_distinct_pairs=int(mask.sum()))
