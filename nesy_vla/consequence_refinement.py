"""Anonymous state refinement driven by measured execution consequences.

Replication initially preserves the parent finite-state policy after summing
child-state probabilities. No ID receives a manually assigned meaning. The
execution decoder exists only as a training objective; deployed actions still
come exclusively from the generated finite-state rule interpreter.
"""
import torch
from torch import nn
from .probabilistic_automaton import ProbabilisticTokenAutomaton
from .factorized_automaton import categorical


def refine_automaton(parent, copies=2):
    if type(copies) is not int or copies<2:
        raise ValueError('At least two anonymous child states required')
    args={k:getattr(parent,k) for k in ['patches','vocabulary','tasks','factors','states','actions','width','rank',
                                      'reset_memory','task_banks','context_bridge']}
    old=args['states'];args['states']*=copies
    child=ProbabilisticTokenAutomaton(**args).to(next(parent.parameters()).device)
    target=child.state_dict();source=parent.state_dict()
    task_offset=int(parent.task_banks)
    axes={'definitions':[1+task_offset], 'definition_basis':[2],
          'transition':[1+task_offset,2+task_offset,4+task_offset],
          'transition_basis':[2,3], 'conclusions':[1+task_offset,2+task_offset],
          'conclusion_basis':[2,3]}
    for name,value in source.items():
        value=value.detach().clone()
        if name in axes:
            for axis in axes[name]:value=value.repeat_interleave(copies,dim=axis)
        elif name=='reconstruct.weight':
            value=value.reshape(value.shape[0],parent.factors,old).repeat_interleave(copies,-1).flatten(1)
        elif name=='predict.0.weight':
            n=parent.factors*old
            expanded=value[:,:n].reshape(value.shape[0],parent.factors,old).repeat_interleave(copies,-1).flatten(1)
            value=torch.cat([expanded,value[:,n:]],dim=1)
        elif name=='inverse.0.weight':
            value=value.reshape(value.shape[0],2,parent.factors,old).repeat_interleave(copies,-1).flatten(1)
        if value.shape!=target[name].shape:
            raise ValueError(f'Unhandled refinement tensor {name}: {value.shape} vs {target[name].shape}')
        target[name]=value
    child.load_state_dict(target,strict=True)
    return child,args


def merge_child_belief(belief,copies=2):
    if belief.shape[-1]%copies:raise ValueError('Invalid state subdivision')
    return belief.reshape(*belief.shape[:-1],belief.shape[-1]//copies,copies).sum(-1)


class ConsequenceDecoder(nn.Module):
    """Training-only prediction from hard equality symbols and task identity."""
    def __init__(self,factors,states,tasks=2,actions=16):
        super().__init__();self.factors=factors;self.states=states;self.tasks=tasks
        self.network=nn.Sequential(nn.Linear(factors*states+tasks,128),nn.GELU(),nn.Linear(128,actions*2))
    def forward(self,observed,tasks):
        if observed.shape[1:]!=(self.factors,self.states):raise ValueError('Wrong discrete state shape')
        detached=observed.detach()
        if not torch.equal(detached,detached.round()) or not torch.equal(detached.sum(-1),torch.ones_like(detached[...,0])):
            raise ValueError('Only hard categorical observations enter the consequence decoder')
        task=torch.nn.functional.one_hot(tasks,self.tasks).to(observed)
        return self.network(torch.cat([observed.flatten(1),task],-1)).reshape(-1,16,2)


def discrete_memory_equalities(belief):
    """Hard anonymous memory IDs for auxiliary training, never visual latents.

The deployed interpreter still marginalizes the full finite-state belief.
This auxiliary branch gives newly duplicated memory conclusions a distinct
learning signal instead of keeping sibling states permanently identical.
"""
    if belief.ndim!=3 or not torch.isfinite(belief).all() or (belief.detach()<0).any():
        raise ValueError('Expected finite categorical memory probabilities')
    if not torch.allclose(belief.detach().sum(-1),torch.ones_like(belief[...,0]),atol=2e-5,rtol=0):
        raise ValueError('Memory probabilities must sum to one')
    return categorical(belief)[1]


def balanced_consequence_loss(logits,labels):
    """Each measured outcome channel balances positive and negative examples."""
    if logits.shape!=labels.shape or logits.shape[1:]!=(16,2):raise ValueError('Expected 16 action results')
    if not ((labels==0)|(labels==1)).all():raise ValueError('Measured labels must be binary')
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    parts=[]
    for channel in range(2):
        for value in [0,1]:
            mask=labels[:,:,channel]==value
            if mask.any():parts.append(loss[:,:,channel][mask].mean())
    return torch.stack(parts).mean()
