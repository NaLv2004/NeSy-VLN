"""Train generated rules through the existing hard backward derivation.

Forward values use the exact argmax successor and integer confidence costs.
The zero-valued surrogate supplies derivatives to rule probabilities, while
the derivative to previous proof support follows the selected hard edge.
No learned actor, value table, state semantics or alternate goal is added.
This module is prepared for a subsequent training revision; v31 is unchanged.
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
    return loss,dict(hard_rule_planning_loss=float(loss.detach()),hard_rule_planning_action_accuracy=float(accuracy.detach()),
        hard_rule_planning_examples=int(valid.sum()),planning_gradients_to_rules=True,soft_runtime_states=False)
