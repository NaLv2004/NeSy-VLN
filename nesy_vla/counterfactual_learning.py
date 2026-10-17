"""Learn grounded Boolean facts from exact discrete rule counterfactuals.

Forward actions and likelihoods are unchanged. Rule parameters use ordinary
likelihood gradients at the actual discrete state. The grounding gradient for
each bit is the measured loss difference between that bit's zero/one states,
holding every other bit and the generated program fixed. No predicate meaning
or desired bit label is prescribed externally.
"""
import torch
from .symbols import DiscreteState


def counterfactual_symbol_bc(logic,program,state,action,probability_floor=1e-9):
    if not isinstance(state,DiscreteState):raise TypeError('Discrete facts required')
    state.validate()
    if state.values.ndim!=1:raise ValueError('One actual state required')
    if not 0<int(action)+1<=logic.a:raise ValueError('Unknown action token')
    codes=state.codes;actual=state.values.detach()
    actual_batch=DiscreteState(codes[None],actual[None])
    probabilities=logic.probabilities_for_states(program,actual_batch)[0]
    nll=-probabilities[action].clamp_min(probability_floor).log()
    with torch.no_grad():
        alternate=actual.expand(len(codes),-1).clone()
        indices=torch.arange(len(codes),device=codes.device)
        alternate[indices,indices]=1-alternate[indices,indices]
        alternatives=DiscreteState(alternate.long(),alternate)
        predictions=logic.probabilities_for_states(program.detach(),alternatives)
        losses=-predictions[:,action].clamp_min(probability_floor).log()
        # For an actual one, the flip is one->zero, hence reverse its sign.
        differences=(losses-nll.detach())*(1-2*actual)
    surrogate=(state.values*differences).sum()
    return nll+(surrogate-surrogate.detach())
