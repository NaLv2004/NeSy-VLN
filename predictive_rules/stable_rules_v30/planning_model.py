"""Train anonymous perception through the same hard backward symbolic proof.

All executable modules and exports are inherited unchanged. Extra objectives
operate only on recorded training transitions, with categorical forward states.
"""
import torch
from torch.nn import functional as F
from joint_model import JointObservedWorld, Config, branch_states
from predictive_nesy.graph_subgoal_model import straight_through
from predictive_nesy.selected_row_objectives import frozen_rule_proof_levels

PlanningObservedWorld=JointObservedWorld


def average(value,mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


def representation_loss(model,p,data,teacher_labels):
    q=model.encode_sequence(p,data['targets'])[0];z=straight_through(q)
    valid=data['action_valid'].float();state_valid=data['state_valid'].float()
    mask=valid*data['targets'][:,1:].ne(data['targets'][:,:-1]).any(-1)
    logits=model.inverse_from_discrete(z[:,:-1],z[:,1:])
    inverse=average(F.cross_entropy(logits.flatten(0,1),data['actions'].flatten(),reduction='none').view_as(valid),mask)
    anchor=average(-q.clamp_min(1e-8).log().gather(-1,teacher_labels[...,None]).squeeze(-1),state_valid)
    marginal=(z*state_valid[...,None]).sum((0,1))/state_valid.sum().clamp_min(1)
    usage=(marginal*marginal.clamp_min(1e-8).log()).sum()
    entropy=average(-(q*q.clamp_min(1e-8).log()).sum(-1),state_valid)
    # The unbalanced teacher action prior must not dominate state learning.
    # The balanced all-action bank supplies the main inverse supervision.
    loss=.1*inverse+.25*anchor+.1*usage
    return loss,dict(inverse_discrete_loss=float(inverse.detach()),
        inverse_discrete_accuracy=float(average((logits.argmax(-1)==data['actions']).float(),mask)),
        inverse_identifiable_examples=float(mask.sum()),representation_anchor=float(anchor.detach()),
        hard_state_perplexity=float((-usage.detach()).exp()),assignment_entropy=float(entropy.detach()),
        online_target_state_agreement=float(average((q.argmax(-1)==teacher_labels).float(),state_valid)))


def perception_proof_loss(model,p,data):
    """Exact corrected categorical source forward; train its visual encoding.

    The table and generated goal are held fixed for this auxiliary gradient.
    For every possible observation ID we calculate the same backward proof
    scores and C correction as execution. A straight-through lookup routes
    teacher-action evidence into perception; no neural action decoder exists.
    """
    q=model.encode_sequence(p,data['targets'])[0];observations=q.argmax(-1)
    b=len(q);device=q.device;h=min(10,32-int(data['start'][0]));index=torch.arange(b,device=device)
    with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
        probabilities=model.rows(p);table=probabilities.argmax(-1)
        goal=F.one_hot(p['subgoal_logits'].argmax(-1).flatten(),512).float()
        levels=frozen_rule_proof_levels(probabilities,goal,h)
        before,evidence,keys,bias,scale=model.correction.parameters_for_program(p['definitions'],p['context_raw'])
    predicted=observations[:,0].detach();losses=[];matches=[];source_equal=[]
    for t in range(h):
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            query=F.normalize(before[index,predicted,None]+evidence+bias[:,None],dim=-1)
            corrected_ids=(scale*torch.einsum('bod,bkd->bok',query,keys)).argmax(-1)
            corrected=corrected_ids.gather(1,observations[:,t,None]).squeeze(1)
            scores=levels[min(10,32-int(data['start'][0])-t)]/(10000*(h+1)*.5)
            targets=data['actions'][:,t,None].expand(-1,512)
            costs=F.cross_entropy(scores.flatten(0,1),targets.flatten(),reduction='none').view(b,512).clamp_max(24.)
            predicted=table[index,corrected,data['actions'][:,t]]
        hard=F.one_hot(corrected,512).float()
        # Scatter the posterior through the actual compiled categorical C row.
        surrogate=torch.zeros_like(q[:,t]).scatter_add(1,corrected_ids,q[:,t])
        state=hard+(surrogate-surrogate.detach())
        assert torch.equal(state.detach(),hard)
        losses.append((state*costs).sum(-1))
        matches.append(scores[index,corrected].argmax(-1)==data['actions'][:,t])
        source_equal.append(corrected==observations[:,t])
    mask=data['action_valid'][:,:h].float()*data['supervised'][:,:h].float()
    loss=average(torch.stack(losses,1),mask)
    return loss,dict(perception_proof_loss=float(loss.detach()),
        perception_proof_action_accuracy=float(average(torch.stack(matches,1).float(),mask)),
        perception_proof_correction_preserved=float(average(torch.stack(source_equal,1).float(),mask)))


def equivalent_action_loss(scores,equivalent):
    # Multiple actions can genuinely produce the identical recorded image.
    # Credit their equivalence class; do not invent distinguishable effects.
    return (torch.logsumexp(scores,-1)-torch.logsumexp(scores.masked_fill(~equivalent,-torch.inf),-1)).mean()


def branch_representation_loss(model,teacher,p,teacher_program,data):
    before,after=branch_states(model,p,data)
    with torch.no_grad():
        tb,ta=branch_states(teacher,teacher_program,data)
        before_ids=tb.argmax(-1);after_ids=ta.argmax(-1)
        equivalent=data['successors'][:,:,None].eq(data['successors'][:,None]).all(-1)
    z0=straight_through(before);z1=straight_through(after);b=len(z0);idx=torch.arange(b,device=z0.device)
    full=model.rows(p);log_full=full.clamp_min(1e-8).log()
    nll=-log_full.gather(-1,after_ids[:,None,:,None].expand(-1,512,-1,1)).squeeze(-1)
    dynamics=(z0[:,:,None]*nll).sum(1).mean()
    exact=log_full[idx,before.argmax(-1)]
    surrogate=torch.einsum('bs,bsak->bak',z0,log_full)
    selected=exact.detach()+(surrogate-surrogate.detach()).float()
    exact_inverse=selected.gather(2,after.argmax(-1)[:,None].expand(-1,17,-1)).transpose(1,2)
    inverse_surrogate=torch.einsum('btk,bak->bta',z1,selected)
    rule_scores=exact_inverse.detach()+(inverse_surrogate-inverse_surrogate.detach()).float()
    rule_inverse=equivalent_action_loss(rule_scores,equivalent)
    source=z0[:,None].expand(-1,17,-1)
    head_scores=model.inverse_from_discrete(source,z1)
    head_inverse=equivalent_action_loss(head_scores,equivalent)
    anchor=-before.clamp_min(1e-8).log().gather(-1,before_ids[:,None]).mean()
    anchor=anchor-after.clamp_min(1e-8).log().gather(-1,after_ids[:,:,None]).mean()
    predicted=full[idx,before.argmax(-1)].argmax(-1)
    head_match=equivalent.gather(-1,head_scores.argmax(-1)[...,None]).float().mean()
    rule_match=equivalent.gather(-1,rule_scores.argmax(-1)[...,None]).float().mean()
    marginal=z1.mean((0,1));usage=(marginal*marginal.clamp_min(1e-8).log()).sum()
    loss=dynamics+head_inverse+.5*rule_inverse+.125*anchor+.1*usage
    return loss,dict(counterfactual_dynamics=float(dynamics.detach()),counterfactual_loss=float(loss.detach()),
        counterfactual_accuracy=float((predicted==after_ids).float().mean()),
        branch_inverse_equivalent_accuracy=float(head_match),branch_rule_inverse_accuracy=float(rule_match),
        branch_rule_inverse_loss=float(rule_inverse.detach()),branch_representation_anchor=float(anchor.detach()),
        branch_hard_state_perplexity=float((-usage.detach()).exp()),
        branch_examples=b*17)
