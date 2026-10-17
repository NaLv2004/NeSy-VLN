"""Auxiliary symbolic objectives; no manually specified predicate meanings."""
import torch
import torch.nn.functional as F


def inverse_transition_logits(predictions,target,discrete=True,temperature=.02):
    """Actions must explain distinct next symbols, not subthreshold confidences."""
    if discrete:
        from .symbols import tokenize
        predictions=tokenize(predictions).values
        target=tokenize(target.detach()).values
    return -(predictions-target.detach()[None]).square().mean(-1)/temperature


def transition_loss(prediction,target,source,confidence,balanced=False):
    """Balance changed/stable learned bits so copying does not dominate training.

    Masks and targets are detached. Source is the exact Boolean state passed to
    the effect interpreter, not the perceptual observation before inference.
    Empty groups are omitted, so an unchanged transition keeps its full weight.
    """
    target=target.detach()
    elementwise=F.binary_cross_entropy(prediction.clamp(1e-5,1-1e-5),target,reduction='none')
    weighted=elementwise*confidence.detach().clamp_min(.2)
    if not balanced:return weighted.mean()
    changed=(source.detach()>=.5)!=(target>=.5)
    pieces=[weighted[group].mean() for group in (changed,~changed) if bool(group.any())]
    return torch.stack(pieces).mean()
