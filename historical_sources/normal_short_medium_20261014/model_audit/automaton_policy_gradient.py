"""Closed-loop policy gradients through generated finite-state programs.

Only integer observations, task IDs and executed-action IDs enter replay.
The auxiliary critic reads detached finite-state beliefs; it cannot select
actions or update the actor through its value-regression objective.
"""
import torch
from torch import nn
import torch.nn.functional as F


CANONICAL_BATCH = 8


class SymbolicValue(nn.Module):
    def __init__(self, factors, states, tasks):
        super().__init__()
        self.tasks = tasks
        self.net = nn.Sequential(nn.Linear(factors * states + tasks, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, belief, task):
        return self.net(torch.cat([belief.detach().flatten(1), F.one_hot(task, self.tasks).to(belief)], -1)).squeeze(-1)


def replay(model, codes, tasks, actions):
    if codes.ndim != 3 or codes.shape[0] != CANONICAL_BATCH:
        raise ValueError('Replay must preserve the original canonical batch of eight trajectories')
    model.validate_codes(codes)
    if actions.dtype != torch.long or actions.shape != codes.shape[:2] or ((actions < 0) | (actions >= model.actions)).any():
        raise TypeError('Expected one executed integer action per observation')
    _, memory = model.initial_state(len(codes), codes.device)
    previous_action = torch.full_like(tasks, model.actions)
    results = []
    for step in range(codes.shape[1]):
        if step % 10 == 0:
            program = model.generate(tasks, codes[:, max(0, step - 10):step + 1], step)
        r = model.step(program, codes[:, step], memory, previous_action, step)
        results.append(r)
        memory = r['state_values']; previous_action = actions[:, step]
    return dict(probabilities=torch.stack([r['probabilities'] for r in results], 1),
                beliefs=torch.stack([r['state_values'] for r in results], 1),
                observed_ids=torch.stack([r['observed_ids'] for r in results], 1))


def advantages(rewards, values, valid, gamma=.99, lam=.95):
    if not (rewards.shape == values.shape == valid.shape) or valid.dtype != torch.bool:
        raise ValueError('Reward, detached value and validity arrays must agree')
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError('A completed episode cannot become active again')
    result = torch.zeros_like(rewards); carry = torch.zeros_like(rewards[:, 0]); next_value = carry.clone()
    for step in reversed(range(rewards.shape[1])):
        active = valid[:, step].to(rewards)
        delta = (rewards[:, step] + gamma * next_value - values[:, step]) * active
        carry = (delta + gamma * lam * carry) * active
        result[:, step] = carry
        next_value = values[:, step] * active
    return result.detach(), (result + values).detach()


def policy_objective(probabilities, actions, old_logp, advantage, valid, clip=.2, entropy_weight=.01):
    logp = probabilities.clamp_min(1e-20).log().gather(-1, actions[..., None]).squeeze(-1)
    ratio = (logp - old_logp).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    entropy = -(probabilities * probabilities.clamp_min(1e-20).log()).sum(-1)
    loss = (-surrogate - entropy_weight * entropy)[valid].mean()
    return loss, {'max_ratio_error': float((ratio.detach()[valid] - 1).abs().max()),
                  'entropy': float(entropy.detach()[valid].mean()),
                  'clip_fraction': float(((ratio.detach() - 1).abs()[valid] > clip).float().mean())}


def configure_trainable(model, arm):
    if arm not in ('joint', 'action_rules_only'):
        raise ValueError('Unknown matched experiment arm')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(arm == 'joint' or name in ('conclusions', 'conclusion_basis', 'weight_logits', 'weight_basis'))
    return [p for p in model.parameters() if p.requires_grad]
