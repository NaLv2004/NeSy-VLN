"""Critic-free grouped policy optimization of the symbolic action distribution."""
import copy
import json

import torch


def validate_groups(episodes, group_size=8):
    """Each group is exactly one environment reset specification, repeated G times."""
    if len(episodes) == 0 or len(episodes) % group_size:
        raise ValueError('Incomplete trajectory group')
    for start in range(0, len(episodes), group_size):
        descriptions = [json.dumps(e, sort_keys=True) for e in episodes[start:start + group_size]]
        if len(set(descriptions)) != 1:
            raise ValueError('GRPO group must share the complete task, start pose and target')


@torch.no_grad()
def group_advantages(rewards, valid, gamma=.99, epsilon=1e-6):
    """Outcome GRPO: one discounted shaped return per rollout, normalized within G=8."""
    if rewards.ndim != 2 or rewards.shape[0] != 8 or rewards.shape != valid.shape:
        raise ValueError('Expected exactly eight complete reward trajectories')
    if valid.dtype != torch.bool or not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError('Invalid episode terminal mask')
    if not torch.isfinite(rewards[valid]).all():
        raise ValueError('Nonfinite environment reward')
    weights = gamma ** torch.arange(rewards.shape[1], device=rewards.device, dtype=torch.float64)
    returns = (torch.where(valid, rewards.double(), 0.) * weights).sum(-1)
    deviation = returns.std(unbiased=False)
    scores = torch.where(deviation < epsilon, torch.zeros_like(returns),
                         (returns - returns.mean()) / deviation.clamp_min(epsilon)).to(rewards)
    advantages = scores[:, None].expand_as(rewards) * valid
    return advantages.detach(), {
        'returns': returns.tolist(), 'return_mean': float(returns.mean()),
        'return_std': float(deviation), 'advantages': scores.tolist(),
        'zero_variance': bool(deviation < epsilon), 'lengths': valid.sum(-1).tolist(),
    }


def group_objective(probabilities, actions, old_logp, advantages, valid, reference,
                    clip=.2, beta=.01, entropy_weight=.01):
    """Mean over actions within each rollout, then equally over the eight rollouts.

    Exact categorical KL(new || frozen starting policy) replaces the noisy
    sampled KL estimator because the complete 17-action distribution is available.
    """
    if probabilities.shape != reference.shape or probabilities.shape[:-1] != valid.shape:
        raise ValueError('Policy/reference/mask shape mismatch')
    if probabilities.shape[0] != 8 or not valid.any(-1).all():
        raise ValueError('Each GRPO group needs eight nonempty rollouts')
    logp = probabilities.clamp_min(1e-20).log()
    chosen = logp.gather(-1, actions[..., None]).squeeze(-1)
    ratio = (chosen - old_logp.detach()).exp()
    advantage = advantages.detach()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    ref_logp = reference.detach().clamp_min(1e-20).log()
    kl = (probabilities * (logp - ref_logp)).sum(-1)
    entropy = -(probabilities * logp).sum(-1)

    def mean_episode(value):
        return ((value * valid).sum(-1) / valid.sum(-1)).mean()

    loss = mean_episode(-surrogate + beta * kl - entropy_weight * entropy)
    return loss, {
        'max_ratio_error': float((ratio.detach()[valid] - 1).abs().max()),
        'clip_fraction': float(((ratio.detach()[valid] - 1).abs() > clip).float().mean()),
        'reference_kl': float(mean_episode(kl.detach())),
        'entropy': float(mean_episode(entropy.detach())),
    }


class ActorSnapshot:
    """Atomic in-memory rollback of actor, optimizer and random states; no critic."""
    def __init__(self, model, optimizer, rng=None):
        self.objects = (model, optimizer)
        self.states = [copy.deepcopy(obj.state_dict()) for obj in self.objects]
        self.rng = rng
        self.python = rng.getstate() if rng is not None else None
        self.cpu = torch.get_rng_state().clone()
        self.cuda = torch.cuda.get_rng_state().clone() if next(model.parameters()).is_cuda else None

    def restore(self):
        for obj, state in zip(self.objects, self.states):
            obj.load_state_dict(state)
        if self.rng is not None:
            self.rng.setstate(self.python)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda)
