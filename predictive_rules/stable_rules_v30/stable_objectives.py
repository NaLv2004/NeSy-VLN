"""Exact hard observed-state objective with a stable learned vocabulary.

Perception, definitions and their causal context are frozen from the best v29
checkpoint. This phase fits the complete rule generator to stationary targets.
Selected-row training is algebraically identical to supervising those rows of
the complete table. Runtime still exports every row and uses the same planner.
"""
import torch
from torch.nn import functional as F
from joint_model import branch_states
from predictive_nesy.selected_row_objectives import frozen_rule_proof_levels
from predictive_nesy.graph_subgoal_model import straight_through


def observed_objective(model,data,p,return_parts=False,target_labels=None):
    b=len(data['task']);device=data['task'].device;h=min(10,32-int(data['start'][0]))
    index=torch.arange(b,device=device)
    with torch.no_grad():
        observations=model.training_targets(p,data['targets'],data['actions']).argmax(-1).squeeze(-1)
        labels=observations if target_labels is None else target_labels
        full=model.rows(p)
        table=full.argmax(-1)
    target_goal=labels[index,data['goal_index']]
    goal_logits=p['subgoal_logits'].squeeze(1)
    goal_loss=F.cross_entropy(goal_logits,target_goal)
    levels=frozen_rule_proof_levels(full,straight_through(goal_logits.softmax(-1)),h)
    state=observations[:,0];premises=[];corrections=[];proof_losses=[];matches=[];correct=[];end_states=[]
    for t in range(h):
        logits=model.correction.selected(p['definitions'],p['context_raw'],state,observations[:,t])
        current=logits.argmax(-1);premises.append(current)
        corrections.append(F.cross_entropy(logits,observations[:,t],reduction='none'))
        depth=min(10,32-int(data['start'][0])-t)
        scores=levels[depth][index,current]/(10000*(h+1)*.5)
        proof_losses.append(F.cross_entropy(scores,data['actions'][:,t],reduction='none'))
        matches.append(scores.argmax(-1)==data['actions'][:,t])
        state=table[index,current,data['actions'][:,t]]
        correct.append(state==labels[:,t+1]);end_states.append(state)
    # Train only the selected premises, without a full-table backward graph.
    rows=model.rows(p,torch.stack(premises,1))
    chosen=rows.gather(2,data['actions'][:,:h,None,None].expand(-1,-1,1,512)).squeeze(2)
    nll=-chosen.gather(-1,labels[:,1:h+1,None]).squeeze(-1).clamp_min(1e-8).log()
    valid=data['action_valid'][:,:h].float();supervised=valid*data['supervised'][:,:h].float()
    def avg(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)
    dynamics=avg(nll,valid);correction=avg(torch.stack(corrections,1),valid)
    proof=avg(torch.stack(proof_losses,1),supervised)
    loss=2*dynamics+goal_loss+proof+correction
    endpoints=torch.stack(end_states,1)[index,data['goal_index'].clamp_max(h)-1]
    output=(loss,dict(loss=float(loss.detach()),observed_conditioned_dynamics=float(dynamics.detach()),
        correction_loss=float(correction.detach()),proof_action_imitation=float(proof.detach()),subgoal_prediction=float(goal_loss.detach()),
        transition_accuracy=float(avg(torch.stack(correct,1).float(),valid)),
        proof_action_accuracy=float(avg(torch.stack(matches,1).float(),supervised)),
        subgoal_exact_accuracy=float((goal_logits.argmax(-1)==target_goal).float().mean()),
        teacher_path_exact_goal=float((endpoints==target_goal).float().mean())),labels)
    if return_parts:return (*output,dict(dynamics=dynamics,proof=proof,correction=correction,goal=goal_loss))
    return output


def branch_representation_loss(model,teacher,p,teacher_program,data):
    """All-action transition likelihood, using the unchanged learned states."""
    with torch.no_grad():
        before,after=branch_states(model,p,data)
        sources=before.argmax(-1);targets=after.argmax(-1)
    probabilities=model.rows(p,sources[:,None])[:,0]
    loss=-probabilities.gather(-1,targets[:,:,None]).squeeze(-1).clamp_min(1e-8).log().mean()
    return loss,dict(counterfactual_dynamics=float(loss.detach()),counterfactual_loss=float(loss.detach()),
        counterfactual_accuracy=float((probabilities.argmax(-1)==targets).float().mean()),
        branch_examples=targets.numel(),stable_observation_labels=True)
