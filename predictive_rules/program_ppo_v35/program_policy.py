"""PPO over generated programs, never over a neural motor-action head.

Two sampled 128-D contexts are the policy decisions. Fixed decoders turn them
into the complete transition table and subgoal; an unchanged hard interpreter
derives every motor action. Keeping the downstream decoder/grounder fixed is
essential: its changes must not escape the PPO likelihood ratio.
"""
import math
import torch
from torch import nn
from raw_model import RawObservedWorld, Config

ACTOR_PREFIXES = ('raw_frame.', 'raw_history.', 'raw_delta.',
                  'goal_frame.', 'goal_history.', 'goal_delta.')


class ProgramPolicy(RawObservedWorld):
    def configure_ppo(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(ACTOR_PREFIXES))
        return [p for p in self.parameters() if p.requires_grad]

    def context_mean(self, task, previous, lengths, visual, base):
        # Dependency setup enables TF32 globally. A256-D density is sensitive
        # to batch-dependent TF32 rounding, so sampling/replay both require FP32.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        index = torch.arange(len(task), device=task.device)
        result = []
        for frame, history, delta in ((self.raw_frame, self.raw_history, self.raw_delta),
                                      (self.goal_frame, self.goal_history, self.goal_delta)):
            values = frame(visual.float().flatten(-2)) + self.action_embed(previous)
            states, _ = history(values.float(), self.task(task).float()[None])
            result.append(base + delta(states[index, lengths-1]))
        return torch.cat(result, -1)

    @torch.no_grad()
    def decision(self, task, history, previous, lengths, *, start, visual, std, stochastic):
        p = self.prepare(task, history, previous, lengths, start=start, boundary_features=visual)
        base = self.context_norm(p['context_raw'])
        mean = self.context_mean(task, previous, lengths, visual, base)
        sample = mean + std * torch.randn_like(mean) if stochastic else mean
        # The rule decoder emits every row, not a selected action or a partial rule.
        p['rule_context'] = sample[:, :self.cfg.hidden]
        p['subgoal_logits'] = self.subgoal(sample[:, self.cfg.hidden:])[:, None]
        p['transitions'] = self.rows(p)
        p['correction_table'] = self.correction.compile(p['definitions'], p['context_raw'])
        value_input = torch.cat([base, visual[:, -1].float().mean(1),
                                torch.nn.functional.one_hot(task, 2).float(),
                                base.new_full((len(task), 1), start/32)], -1)
        record = dict(task=task.cpu(), previous=previous.cpu(), lengths=lengths.cpu(),
                      visual=visual.cpu(), base=base.cpu(), mean=mean.cpu(), sample=sample.cpu(),
                      value_input=value_input.cpu(), logp=normal_logp(sample, mean, std).cpu())
        return p, record


def normal_logp(sample, mean, std):
    # Sum over the full program decision. Averaging dimensions is NOT a density.
    return (-.5*((sample-mean)/std).square()-math.log(std)-.5*math.log(2*math.pi)).sum(-1)


def exact_kl(old, new, std):
    return .5*((old-new)/std).square().sum(-1)


class ProgramValue(nn.Module):
    """Training-only scalar baseline. Not exported to the symbolic controller."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(259), nn.Linear(259,256), nn.Tanh(),
                                     nn.Linear(256,128), nn.Tanh(), nn.Linear(128,1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values):
        return self.network(values).squeeze(-1)


def replay_batch(records, device='cuda'):
    width = max(int(r['lengths'][0]) for r in records)
    count = len(records)
    previous = torch.full((count,width),17,dtype=torch.long,device=device)
    visual = torch.zeros(count,width,49,128,dtype=torch.float16,device=device)
    for i,r in enumerate(records):
        size = int(r['lengths'][0])
        previous[i,:size] = r['previous'][0].to(device)
        visual[i,:size] = r['visual'][0].to(device)
    result = {key:torch.cat([r[key] for r in records]).to(device)
              for key in ['task','lengths','base','mean','sample','value_input','logp']}
    result.update(previous=previous,visual=visual)
    return result


def semi_markov_advantages(segments, gamma=.99, lam=.95):
    """Discount by actual executed actions, rather than by program count."""
    following_value = 0.
    following_advantage = 0.
    for segment in reversed(segments):
        rewards = segment['rewards']
        # A rejected proof executes no motor action but still gets its own cost.
        duration = max(1,segment['executed'])
        discounted = -segment['generation_cost'] + sum(gamma**i*r for i,r in enumerate(rewards))
        discount = gamma**duration
        delta = discounted + discount*following_value - segment['value']
        advantage = delta + (gamma*lam)**duration*following_advantage
        segment['advantage'] = advantage
        segment['return'] = advantage + segment['value']
        following_value = segment['value']
        following_advantage = advantage

