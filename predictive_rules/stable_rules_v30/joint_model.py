"""Jointly learned anonymous visual states; runtime stays categorical."""
import torch
from torch import nn
from torch.nn import functional as F
from adaptive_model import AdaptiveObservedWorld,Config
from predictive_nesy.graph_subgoal_model import straight_through


class JointObservedWorld(AdaptiveObservedWorld):
    def __init__(self,config=None):
        super().__init__(config)
        # Training-only inverse dynamics sees two one-hot IDs. No image,
        # visual embedding, definition embedding, or continuous-state bypass.
        self.inverse_discrete=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,17))

    def inverse_from_discrete(self,before,after):
        if before.shape!=after.shape or before.shape[-1]!=512:
            raise ValueError('Inverse dynamics requires two categorical states')
        for state in (before,after):
            detached=state.detach()
            if not torch.all((detached==0)|(detached==1)) or not torch.all(detached.sum(-1)==1):
                raise ValueError('Inverse dynamics cannot consume a soft state or continuous image feature')
        return self.inverse_discrete(torch.cat([before,after],-1))


def representation_loss(model,prepared,data,teacher_labels):
    """Data-derived predictive abstraction with slowly moving target labels.

    All transitions are hard one-hot in the forward pass. The straight-through
    surrogate permits perception/definition learning. Future images are label
    sources during training, never runtime program inputs.
    """
    q=model.encode_sequence(prepared,data['targets'])[0]
    z=straight_through(q)
    valid=data['action_valid'].float()
    changed=data['targets'][:,1:].ne(data['targets'][:,:-1]).any(-1).float()
    inverse_mask=valid*changed
    logits=model.inverse_from_discrete(z[:,:-1],z[:,1:])
    nll=F.cross_entropy(logits.flatten(0,1),data['actions'].flatten(),reduction='none').view_as(valid)
    inverse=(nll*inverse_mask).sum()/inverse_mask.sum().clamp_min(1)
    anchor=-q.clamp_min(1e-8).log().gather(-1,teacher_labels[...,None]).squeeze(-1)
    state_valid=data['state_valid'].float()
    anchor=(anchor*state_valid).sum()/state_valid.sum().clamp_min(1)
    average=(z*state_valid[...,None]).sum((0,1))/state_valid.sum().clamp_min(1)
    # Marginal usage regularization discourages collapsing every image into a
    # single ID without assigning any hand-written meaning to an ID.
    usage=(average*average.clamp_min(1e-8).log()).sum()
    hard_average=(z.detach()*state_valid[...,None]).sum((0,1))/state_valid.sum().clamp_min(1)
    perplexity=(-(hard_average*hard_average.clamp_min(1e-8).log()).sum()).exp()
    assignment_entropy=(-(q*q.clamp_min(1e-8).log()).sum(-1)*state_valid).sum()/state_valid.sum().clamp_min(1)
    loss=inverse+.25*anchor+.02*usage+.01*assignment_entropy
    return loss,dict(inverse_discrete_loss=float(inverse.detach()),
        inverse_discrete_accuracy=float(((logits.argmax(-1)==data['actions']).float()*inverse_mask).sum()/inverse_mask.sum().clamp_min(1)),
        inverse_identifiable_examples=float(inverse_mask.sum()),representation_anchor=float(anchor.detach()),
        assignment_entropy=float(assignment_entropy.detach()),
        hard_state_perplexity=float(perplexity),
        online_target_state_agreement=float(((q.argmax(-1)==teacher_labels).float()*state_valid).sum()/state_valid.sum().clamp_min(1)))


def branch_states(model,p,data):
    c=model.cfg;b=len(data['task'])
    sequence,memory,previous,codes=model.encode_sequence(p,data['prefix_codes'])
    with torch.autocast(device_type=codes.device.type,enabled=False):
        future=model.observation(model.embed(data['successors'])).flatten(0,1)
        updated,feature,_=model.advance(memory.repeat_interleave(c.actions,0),previous.repeat_interleave(c.actions,0),
            codes.repeat_interleave(c.actions,0),future,data['successors'].flatten(0,1))
        projected=F.normalize(feature+model.memory_projection(updated),dim=-1)
        definitions=F.normalize(p['definitions'],dim=-1).repeat_interleave(c.actions,0)
        logits=8*torch.einsum('bd,bkd->bk',projected,definitions).float()
    return sequence[:,-1],logits.softmax(-1).view(b,c.actions,c.states)


def branch_representation_loss(model,teacher,p,teacher_program,data):
    before,after=branch_states(model,p,data)
    with torch.no_grad():
        target_before,target_after=branch_states(teacher,teacher_program,data)
        before_ids=target_before.argmax(-1);after_ids=target_after.argmax(-1)
    z0=straight_through(before);z1=straight_through(after)
    # Hard source row forward, expected row-NLL straight-through derivative.
    # The successor remains a hard EMA categorical target, not an RGB target.
    full=model.rows(p)
    nll=-full.gather(-1,after_ids[:,None,:,None].expand(-1,512,-1,1)).squeeze(-1).clamp_min(1e-8).log()
    dynamics=(z0[:,:,None]*nll).sum(1).mean()
    source=z0[:,None].expand(-1,17,-1)
    inverse_logits=model.inverse_from_discrete(source,z1)
    actions=torch.arange(17,device=before.device)[None].expand(len(before),-1)
    changed=data['successors'].ne(data['prefix_codes'][:,-1,None]).any(-1).float()
    inverse=F.cross_entropy(inverse_logits.flatten(0,1),actions.flatten(),reduction='none').view_as(changed)
    inverse=(inverse*changed).sum()/changed.sum().clamp_min(1)
    anchor=-before.clamp_min(1e-8).log().gather(-1,before_ids[:,None]).mean()
    anchor=anchor-after.clamp_min(1e-8).log().gather(-1,after_ids[:,:,None]).mean()
    predicted=full[torch.arange(len(before),device=before.device),before.argmax(-1)].argmax(-1)
    loss=dynamics+.5*inverse+.125*anchor
    return loss,dict(counterfactual_dynamics=float(dynamics.detach()),counterfactual_loss=float(loss.detach()),
        counterfactual_accuracy=float((predicted==after_ids).float().mean()),
        branch_inverse_discrete_accuracy=float(((inverse_logits.argmax(-1)==actions).float()*changed).sum()/changed.sum().clamp_min(1)),
        branch_inverse_examples=float(changed.sum()),branch_representation_anchor=float(anchor.detach()))
