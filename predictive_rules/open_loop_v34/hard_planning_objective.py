"""Train generated rules through the existing hard backward derivation.

Forward values use the exact argmax successor and integer confidence costs.
The zero-valued surrogate supplies derivatives to rule probabilities, while
the derivative to previous proof support follows the selected hard edge.
No learned actor, value table, state semantics or alternate goal is added.
Multi-step state supervision uses only the initial observed state as input.
"""
import math
import torch
from torch.nn import functional as F
from predictive_nesy.graph_subgoal_model import straight_through


def hard_rule_proof_levels(probabilities,goal,horizon):
    if probabilities.ndim!=4 or probabilities.shape[-1]!=probabilities.shape[1]:
        raise ValueError('Expected complete categorical rule distributions')
    ids=probabilities.argmax(-1);shape=ids.shape
    scale=10000;factor=horizon+1
    peak=probabilities.gather(-1,ids[...,None]).squeeze(-1)
    log_peak=peak.clamp_min(1e-8).log()
    # Match the existing planner's rounded log-confidence forward values.
    lattice=torch.round(peak.detach().double().log()*scale).float()*factor-1
    confidence=lattice+(log_peak-log_peak.detach())*(scale*factor)
    unreachable=math.ceil((horizon+1)*math.log(probabilities.shape[-1])+1)*scale*factor
    terminal=-unreachable*(1-goal);support=terminal;levels=[None]
    zero=probabilities-probabilities.detach()
    def lookup(values):
        exact=values.gather(1,ids.flatten(1)).reshape(shape)
        # The first term has the exact discrete value and support derivative.
        # The second term is numerically zero and trains only the rule logits.
        surrogate=torch.einsum('bsak,bk->bsa',zero,values.detach())
        return exact+surrogate
    direct=lookup(terminal)+confidence
    for depth in range(1,horizon+1):
        if depth==1:scores=direct
        else:scores=torch.cat([direct[:,:,:1],(lookup(support)+confidence)[:,:,1:]],-1)
        levels.append(scores);support=scores.amax(-1)
    return levels


def hard_open_loop_objective(model,program,data,probabilities,observed,horizon):
    """Predict hard symbolic futures without inserting any future observation.

    Grounded future IDs are detached supervision targets only. The recurrence
    sees an initial corrected integer ID, teacher actions, and generated T.
    Every subsequent premise is its own previous argmax prediction.
    """
    index=torch.arange(len(observed),device=observed.device)
    with torch.no_grad():
        initial=model.correction.selected(program['definitions'],program['context_raw'],
            observed[:,0],observed[:,0]).argmax(-1)
    state=F.one_hot(initial,probabilities.shape[-1]).float()
    valid=data['action_valid'][:,:horizon].float()
    losses=[];correct=[];predicted=[]
    with torch.autocast(device_type=probabilities.device.type,enabled=False):
        for t in range(horizon):
            actions=data['actions'][:,t]
            # This is exactly the conditional distribution of the selected
            # hard T row; only its argmax enters the next symbolic step.
            selected=probabilities[index,state.detach().argmax(-1),actions]
            # Zero forward value, derivative only to the previous hard state.
            candidate_rows=probabilities[index,:,actions].detach()
            through_state=torch.einsum('bs,bsk->bk',state-state.detach(),candidate_rows)
            following=selected+through_state
            target=observed[:,t+1]
            losses.append(-following.gather(-1,target[:,None]).squeeze(-1).clamp_min(1e-8).log())
            following_state=straight_through(following)
            ids=following_state.detach().argmax(-1)
            predicted.append(ids);correct.append(ids==target)
            state=torch.where(valid[:,t,None].bool(),following_state,state)
    normalizer=valid.sum().clamp_min(1)
    loss=(torch.stack(losses,1)*valid).sum()/normalizer
    accuracy=(torch.stack(correct,1).float()*valid).sum()/normalizer
    endpoint_index=data['goal_index'].clamp(1,horizon)-1
    endpoint=torch.stack(predicted,1)[index,endpoint_index]
    endpoint_target=observed[index,data['goal_index'].clamp_max(horizon)]
    return loss,dict(open_loop_dynamics_loss=float(loss.detach()),
        open_loop_transition_accuracy=float(accuracy.detach()),
        open_loop_teacher_endpoint_accuracy=float(endpoint.eq(endpoint_target).float().mean()),
        open_loop_examples=int(valid.sum()),open_loop_future_observation_corrections=0)


def planning_rule_objective(model,data,program):
    """Teacher-action supervision routed through the complete hard proof.

    A smaller independently sampled demonstration batch can bound the dense
    full-table gradient memory. Every action score remains a backward proof
    score to the model-generated hard goal, never a neural action-head score.
    """
    h=min(10,32-int(data['start'][0]));index=torch.arange(len(data['task']),device=data['task'].device)
    with torch.no_grad():
        observed=model.training_targets(program,data['targets'],data['actions']).argmax(-1).squeeze(-1)
    probabilities=model.rows(program)
    goal=straight_through(program['subgoal_logits'].squeeze(1).softmax(-1))
    levels=hard_rule_proof_levels(probabilities,goal,h)
    table=probabilities.detach().argmax(-1);predicted=observed[:,0]
    losses=[];correct=[]
    for t in range(h):
        with torch.no_grad():
            current=model.correction.selected(program['definitions'],program['context_raw'],predicted,observed[:,t]).argmax(-1)
            predicted=table[index,current,data['actions'][:,t]]
        depth=min(10,32-int(data['start'][0])-t)
        scores=levels[depth][index,current]/(10000*(h+1)*.5)
        losses.append(F.cross_entropy(scores,data['actions'][:,t],reduction='none'))
        correct.append(scores.argmax(-1).eq(data['actions'][:,t]))
    valid=data['action_valid'][:,:h].float()*data['supervised'][:,:h].float()
    loss=(torch.stack(losses,1)*valid).sum()/valid.sum().clamp_min(1)
    accuracy=(torch.stack(correct,1).float()*valid).sum()/valid.sum().clamp_min(1)
    open_loop,open_metrics=hard_open_loop_objective(model,program,data,probabilities,observed,h)
    metrics=dict(hard_rule_planning_loss=float(loss.detach()),hard_rule_planning_action_accuracy=float(accuracy.detach()),
        hard_rule_planning_examples=int(valid.sum()),planning_gradients_to_rules=True,soft_runtime_states=False)
    metrics.update(open_metrics)
    # The caller weights this combined objective by0.25: backward-proof
    # imitation therefore remains0.25 and multi-step dynamics has weight0.5.
    return loss+2.*open_loop,metrics
