"""Discrete predictive learning with hard forward states and learned labels."""
import torch
from torch.nn import functional as F
from raw_model import branch_states
from predictive_nesy.graph_subgoal_model import straight_through
from stable_objectives import observed_objective
from planning_model import representation_loss,perception_proof_loss,equivalent_action_loss


def rows_for_states(model,p,states):
    # One-hot forward selects exactly one definition; straight-through
    # derivatives train the categorical grounder rather than a continuous
    # replacement state. The full hard table is still exported at runtime.
    hard=states.detach()
    if not torch.all((hard==0)|(hard==1)) or not torch.all(hard.sum(-1)==1):raise ValueError('Noncategorical training state')
    definitions=p['definitions'];source=states@definitions
    evidence=model.rule_evidence(source,p['rule_context'],p['rule_visual'])
    transition=model.definition_rules
    keys=F.normalize(transition.key(definitions),dim=-1)
    x=transition.premise(source)[:,:,None]+transition.action.weight[None,None]+transition.context(p['rule_context'])[:,None,None]
    query=F.normalize(transition.query(F.gelu(x+evidence[:,:,None])),dim=-1)
    return (transition.log_scale.exp().clamp(1,64)*torch.einsum('bsad,bkd->bsak',query,keys).float()).softmax(-1)


def branch_objective(model,teacher,p,tp,data):
    before,after=branch_states(model,p,data)
    with torch.no_grad():
        tb,ta=branch_states(teacher,tp,data);bids=tb.argmax(-1);targets=ta.argmax(-1)
        visual=data['successor_visual'].flatten(-2)
        equivalent=visual[:,:,None].eq(visual[:,None]).all(-1)
    z0=straight_through(before);z1=straight_through(after)
    probabilities=rows_for_states(model,p,z0[:,None])[:,0]
    dynamics=-probabilities.gather(-1,targets[:,:,None]).squeeze(-1).clamp_min(1e-8).log().mean()
    logits=model.inverse_from_discrete(z0[:,None].expand(-1,17,-1),z1)
    inverse=equivalent_action_loss(logits,equivalent)
    # Ask the generated transition rules themselves which action explains each
    # actual successor ID. The independent inverse head cannot solve this loss
    # on behalf of the executable rules.
    log_rules=probabilities.clamp_min(1e-8).log()
    actual_after=z1.detach().argmax(-1)
    exact=log_rules.gather(2,actual_after[:,None].expand(-1,17,-1)).transpose(1,2)
    # Exact categorical lookup forward; this zero-valued term supplies the
    # representation gradient to the encoded successor. No soft runtime state.
    label_derivative=torch.einsum('btk,bak->bta',z1-z1.detach(),log_rules.detach())
    rule_inverse_scores=exact+label_derivative
    rule_inverse=equivalent_action_loss(rule_inverse_scores,equivalent)
    rule_inverse_match=equivalent.gather(-1,rule_inverse_scores.argmax(-1)[...,None]).float().mean()
    anchor=-before.clamp_min(1e-8).log().gather(-1,bids[:,None]).mean()-after.clamp_min(1e-8).log().gather(-1,targets[:,:,None]).mean()
    marginal=z1.mean((0,1));usage=(marginal*marginal.clamp_min(1e-8).log()).sum()
    loss=dynamics+inverse+.5*rule_inverse+.125*anchor+.1*usage
    return loss,dict(counterfactual_dynamics=float(dynamics.detach()),counterfactual_accuracy=float(probabilities.argmax(-1).eq(targets).float().mean()),
        branch_inverse_accuracy=float(equivalent.gather(-1,logits.argmax(-1)[...,None]).float().mean()),
        branch_state_perplexity=float((-usage.detach()).exp()),branch_examples=targets.numel(),
        branch_anchor=float(anchor.detach()),branch_loss=float(loss.detach()),
        branch_rule_inverse_loss=float(rule_inverse.detach()),
        branch_rule_inverse_accuracy=float(rule_inverse_match),
        branch_program_start=data['start'],branch_program_age=data['offset'],
        branch_observation_step=data['start']+data['offset'])


def demonstration_objective(model,teacher,data,p,tp):
    with torch.no_grad():target=teacher.training_targets(tp,data['targets'],data['actions']).argmax(-1).squeeze(-1)
    loss,metrics,labels=observed_objective(model,data,p,target_labels=target)
    # Preserve the same hard backward proof used by the executor. Its
    # perception surrogate and inverse-dynamics objective now see raw features.
    representation,rm=representation_loss(model,p,data,target)
    proof,pm=perception_proof_loss(model,p,data)
    loss=loss+representation+.25*proof;metrics.update(rm);metrics.update(pm)
    return loss,metrics,labels
