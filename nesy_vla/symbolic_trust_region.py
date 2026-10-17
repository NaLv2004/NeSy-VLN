"""Bound optimizer changes without modifying symbolic action inference."""
import copy
import torch


@torch.no_grad()
def policy_drift(old, new, actions, valid, clip=.2):
    """Additive KL and clipping statistics, weighted by actual valid actions."""
    if old.shape != new.shape or old.shape[:-1] != valid.shape or actions.shape != valid.shape:
        raise ValueError('Policy and trajectory shapes disagree')
    if valid.dtype != torch.bool or actions.dtype != torch.long:
        raise TypeError('Expected a Boolean terminal mask and integer action IDs')
    if not valid.any():
        raise ValueError('An empty rollout cannot constrain an update')
    p=old[valid].double();q=new[valid].double();a=actions[valid]
    if not torch.isfinite(p).all() or (p<0).any() or not torch.allclose(p.sum(-1),torch.ones_like(p[:,0]),atol=2e-5,rtol=0):
        raise ValueError('Invalid behavior-policy probabilities')
    if not torch.isfinite(q).all() or (q<0).any() or not torch.allclose(q.sum(-1),torch.ones_like(q[:,0]),atol=2e-5,rtol=0):
        return p.new_tensor([float('inf'),0.,len(p)])
    lp=p.clamp_min(1e-20).log();lq=q.clamp_min(1e-20).log()
    kl=(p*(lp-lq)).sum(-1).sum().clamp_min(0.)
    log_ratio=(lq-lp).gather(1,a[:,None]).squeeze(1)
    clipped=((log_ratio.exp()-1).abs()>clip).sum()
    return torch.stack([kl,clipped.to(kl),kl.new_tensor(len(p))])


class UpdateSnapshot:
    """Restore rejected actor/critic updates, moments and random generators."""
    def __init__(self, model, optimizer, critic, value_optimizer, rng=None):
        self.objects=(model,optimizer,critic,value_optimizer)
        self.states=[copy.deepcopy(obj.state_dict()) for obj in self.objects]
        self.rng=rng;self.python=rng.getstate() if rng is not None else None
        self.cpu=torch.get_rng_state().clone()
        self.cuda=torch.cuda.get_rng_state().clone() if next(model.parameters()).is_cuda else None

    def restore(self):
        for obj,state in zip(self.objects,self.states):obj.load_state_dict(state)
        if self.rng is not None:self.rng.setstate(self.python)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:torch.cuda.set_rng_state(self.cuda)
