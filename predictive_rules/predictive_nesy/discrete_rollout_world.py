"""Discrete visual dynamics with training-only multi-step gradient propagation.

The forward state is always a categorical symbol, including during training.
Soft derivatives are a straight-through training surrogate. No actual future
frame is accepted by the rollout method. This remains a world component, not
an executable neural policy or a replacement for the exact symbolic planner.
"""
import torch
from torch.nn import functional as F
from .discrete_visual_world import DiscreteVisualWorld


class DiscreteRolloutWorld(DiscreteVisualWorld):
    def from_distribution(self, state, action):
        if state.shape[1:] != (self.cfg.patches, self.cfg.vocabulary):
            raise ValueError('Expected categorical visual symbols')
        exact = self.symbol(state.argmax(-1))
        if torch.is_grad_enabled():
            surrogate = state.to(self.symbol.weight.dtype) @ self.symbol.weight
            x = exact.detach() + (surrogate-surrogate.detach()).to(exact.dtype)
        else:
            x = exact
        x = x + self.position + self.action(action)[:, None]
        return self.output(self.norm(self.encoder(x)))

    def rollout(self, initial, actions):
        if initial.dtype != torch.long or actions.ndim != 2:
            raise ValueError('Rollout accepts initial symbol IDs and action IDs only')
        state = F.one_hot(initial, self.cfg.vocabulary).float()
        outputs = []
        for action in actions.unbind(1):
            logits = self.from_distribution(state, action)
            outputs.append(logits)
            soft = logits.float().softmax(-1)
            hard = F.one_hot(soft.argmax(-1), self.cfg.vocabulary).to(soft.dtype)
            state = hard + (soft-soft.detach()) if torch.is_grad_enabled() else hard
        return torch.stack(outputs, 1)

    def forward(self, before, actions, initial=None, sequence_actions=None):
        one_step = super().forward(before, actions)
        if initial is None:
            return one_step
        return one_step, self.rollout(initial, sequence_actions)
