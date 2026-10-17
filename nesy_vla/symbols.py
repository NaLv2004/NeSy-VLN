"""The enforced boundary between visual grounding and symbolic execution."""
from dataclasses import dataclass
import torch


class _HardBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx,probabilities):return (probabilities>=.5).to(probabilities.dtype)

    @staticmethod
    def backward(ctx,gradient):return gradient


class _HardCategory(torch.autograd.Function):
    @staticmethod
    def forward(ctx,probabilities):
        return torch.zeros_like(probabilities).scatter_(-1,probabilities.argmax(-1,keepdim=True),1.)

    @staticmethod
    def backward(ctx,gradient):return gradient


@dataclass(frozen=True)
class DiscreteState:
    codes: torch.Tensor
    values: torch.Tensor

    def validate(self):
        if self.codes.dtype!=torch.long or self.codes.shape!=self.values.shape:
            raise TypeError('Reasoning accepts integer predicate codes with matching straight-through values')
        if not torch.all((self.codes==0)|(self.codes==1)):
            raise ValueError('Predicate values must be discrete Boolean symbols')
        if not torch.equal(self.values.detach(),self.codes.to(self.values.dtype)):
            raise ValueError('Continuous visual values cannot cross the symbolic boundary')
        return self


def tokenize(probabilities):
    if not torch.isfinite(probabilities).all():raise FloatingPointError('Non-finite perceptual symbol probabilities')
    values=_HardBinary.apply(probabilities)
    return DiscreteState(values.detach().long(),values).validate()


def tokenize_categories(logits,groups):
    """Competing anonymous learned categories, represented as discrete facts.

    Grouping specifies an alphabet structure only. No category is assigned a
    human meaning, motor command, object identity or spatial relation here.
    Used by isolated learning probes; the default navigation path is unchanged.
    """
    if type(groups) is not int or groups<1 or logits.ndim<1 or logits.shape[-1]%groups:
        raise ValueError('A positive group count must divide the learned vocabulary')
    if logits.shape[-1]//groups<2:raise ValueError('Each learned group needs at least two categories')
    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite category logits')
    probabilities=logits.reshape(*logits.shape[:-1],groups,-1).softmax(-1)
    values=_HardCategory.apply(probabilities).reshape_as(logits)
    return DiscreteState(values.detach().long(),values).validate()


def replay_observed_tokens(fresh,recorded_codes):
    """Replay the actual discrete observation, retaining the grounding STE gradient.

    PPO is conditioned on tokens observed during sampling. Batch-dependent
    encoder rounding must not silently replace that observation on replay.
    """
    if not isinstance(fresh,DiscreteState):raise TypeError('Expected grounded discrete symbols')
    fresh.validate()
    codes=torch.as_tensor(recorded_codes,device=fresh.codes.device)
    if codes.dtype!=torch.long or codes.shape!=fresh.codes.shape:
        raise TypeError('Recorded observations must be matching integer symbol tokens')
    values=codes.to(fresh.values.dtype)+(fresh.values-fresh.values.detach())
    return DiscreteState(codes,values).validate()
