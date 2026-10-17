"""Supervise actual future visual histories, never runtime state corrections."""
import torch
from torch.nn import functional as F
from .raw_visual_objectives import objective
from .graph_subgoal_model import straight_through
from .graph_subgoal_training import transition
from .subgoal_data import require_state_only


def transition_objective(model, batch,*,state_separation_weight=0.,program=None):
    require_state_only(batch)
    b=len(batch['task']);c=model.cfg;device=batch['task'].device
    lengths=torch.full((b,),batch['history'].shape[1],device=device,dtype=torch.long)
    p=program if program is not None else model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=batch['start'])
    observed,memory,previous,codes=model.encode_sequence(p,batch['prefix_codes'])
    before=straight_through(observed[:,-1])
    predicted=torch.einsum('bs,bsak->bak',before,p['transitions']).float()
    future_features=model.observation(model.embed(batch['successors'])).flatten(0,1)
    new_memory,new_features,_=model.advance(memory.repeat_interleave(c.actions,0),previous.repeat_interleave(c.actions,0),
        codes.repeat_interleave(c.actions,0),future_features,batch['successors'].flatten(0,1))
    features=F.normalize(new_features+model.memory_projection(new_memory),dim=-1)
    definitions=F.normalize(p['definitions'],dim=-1).repeat_interleave(c.actions,0)
    posterior=(8*torch.einsum('bd,bkd->bk',features,definitions).float()).softmax(-1).reshape(b,c.actions,c.states)
    hard=straight_through(posterior)
    dynamics=-(hard.detach()*predicted.clamp_min(1e-8).log()).sum(-1).mean()
    alignment=-(posterior*predicted.detach().clamp_min(1e-8).log()).sum(-1).mean()
    state=straight_through(observed[:,0])
    for t in range(batch['offset']):state=straight_through(transition(p,state,batch['prefix_actions'][:,t]))
    propagated=torch.einsum('bs,bsak->bak',state,p['transitions']).float()
    open_loop=-(hard.detach()*propagated.clamp_min(1e-8).log()).sum(-1).mean()
    reconstruction=F.cross_entropy(model.reconstruct_codes(p,hard).flatten(0,2),batch['successors'].flatten())
    loss=dynamics+.5*open_loop+.1*alignment+.5*reconstruction
    metrics=dict(counterfactual_dynamics=float(dynamics.detach()),counterfactual_open_loop=float(open_loop.detach()),
        counterfactual_accuracy=float((predicted.argmax(-1)==hard.argmax(-1)).float().mean()))
    if state_separation_weight:
        from .anonymous_separation import separation
        extra,diagnostic=separation(posterior,batch['successors'])
        loss=loss+state_separation_weight*extra
        metrics.update({('counterfactual_'+k):v for k,v in diagnostic.items()},counterfactual_separation=float(extra.detach()))
    metrics['counterfactual_loss']=float(loss.detach())
    return loss,metrics
