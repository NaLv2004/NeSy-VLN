"""Inverse supervision that never receives the action-bearing recurrent state."""
import torch
import torch.nn.functional as F
from .factorized_automaton import categorical


def observation_equalities(model,program,codes,step):
    """Ground one integer observation under a common generated definition set."""
    model.validate_codes(codes)
    if codes.ndim!=2:raise ValueError('Expected one integer grid per observation')
    if not program.start<=step<program.start+10:raise ValueError('Expired program')
    features=model.observation(model.embed(codes)).reshape(-1,model.factors,model.width)
    logits=torch.einsum('bfd,bfkd->bfk',F.normalize(features,dim=-1),F.normalize(program.definitions,dim=-1))*5.
    return categorical(logits.softmax(-1))[1]


def observation_inverse_logits(model,program,before_codes,after_codes,step):
    """Auxiliary label prediction from two hard observation states only.

    Both frames use the current program, even across a ten-action revision.
    No executed-action token, transition table, memory belief or geometry is
    an input. This helper never supplies deployed action probabilities.
    """
    before=observation_equalities(model,program,before_codes,step)
    after=observation_equalities(model,program,after_codes,step)
    return model.inverse_action(before,after)
