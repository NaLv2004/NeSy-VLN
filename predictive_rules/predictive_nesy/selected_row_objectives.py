"""Same separated supervision, without dense unused-row backward graphs."""
import math
import torch
from torch.nn import functional as F
from .graph_subgoal_model import straight_through
from .graph_subgoal_training import mean
from .subgoal_data import require_state_only


def frozen_rule_proof_levels(probabilities,goal,horizon):
    """Exact categorical proof with gradients only to the generated subgoal.

    Since the rule probabilities are detached, a one-hot rule multiplication
    and an index gather have identical derivatives with respect to goal support.
    No rule gradient, learned value function or alternative goal is introduced.
    """
    assert not probabilities.requires_grad
    ids=probabilities.argmax(-1);shape=ids.shape
    scale=10000;factor=horizon+1
    confidence=torch.round(probabilities.max(-1).values.double().log()*scale).float()*factor-1
    unreachable=math.ceil((horizon+1)*math.log(probabilities.shape[-1])+1)*scale*factor
    terminal=-unreachable*(1-goal);support=terminal;levels=[None]
    def lookup(x):return x.gather(1,ids.flatten(1)).reshape(shape)
    direct=lookup(terminal)+confidence
    for depth in range(1,horizon+1):
        if depth==1:q=direct
        else:q=torch.cat([direct[:,:,:1],(lookup(support)+confidence)[:,:,1:]],-1)
        levels.append(q);support=q.amax(-1)
    return levels


def demonstration_objective(model,batch,prepared,*,return_parts=False):
    require_state_only(batch);b=len(batch['task']);device=batch['task'].device;h=min(10,32-int(batch['start'][0]))
    assert torch.all(batch['start']==prepared['start'])
    with torch.no_grad():labels=model.training_targets(prepared,batch['targets'],batch['actions']).argmax(-1).squeeze(-1)
    valid=batch['action_valid'].float();good=((batch['supervised']|~batch['action_valid']).all(-1)).float()
    rows=model.rows(prepared,labels[:,:-1])
    teacher_next=rows.gather(2,batch['actions'][:,:,None,None].expand(-1,-1,1,model.cfg.states)).squeeze(2)
    nll=-teacher_next.gather(-1,labels[:,1:,None]).squeeze(-1).clamp_min(1e-8).log()
    dynamics=mean(nll,valid)
    target=labels[torch.arange(b,device=device),batch['goal_index']]
    goal_logits=prepared['subgoal_logits'].squeeze(1)
    goal_loss=mean(F.cross_entropy(goal_logits,target,reduction='none'),good)
    goal=straight_through(goal_logits.softmax(-1))
    with torch.no_grad():
        full=model.rows(prepared);table=full.argmax(-1);state=labels[:,0];premises=[];predicted=[]
        for t in range(10):
            premises.append(state);state=table[torch.arange(b,device=device),state,batch['actions'][:,t]];predicted.append(state)
        predicted=torch.stack(predicted,1)
    levels=frozen_rule_proof_levels(full,goal,h);losses=[];matches=[]
    for t in range(h):
        scores=levels[h-t][torch.arange(b,device=device),premises[t]]/(10000*(h+1)*.5)
        losses.append(F.cross_entropy(scores,batch['actions'][:,t],reduction='none'))
        matches.append(scores.argmax(-1)==batch['actions'][:,t])
    mask=valid[:,:h]*batch['supervised'][:,:h].float()*good[:,None]
    planning=mean(torch.stack(losses,1),mask);loss=2*dynamics+goal_loss+planning
    metrics=dict(loss=float(loss.detach()),one_step_dynamics=float(dynamics.detach()),subgoal_prediction=float(goal_loss.detach()),
        proof_action_imitation=float(planning.detach()),one_step_accuracy=float(mean((teacher_next.argmax(-1)==labels[:,1:]).float(),valid)),
        transition_accuracy=float(mean((predicted==labels[:,1:]).float(),valid)),subgoal_exact_accuracy=float(mean((goal.argmax(-1)==target).float(),good)),
        teacher_path_exact_goal=float(mean((predicted[torch.arange(b,device=device),batch['goal_index']-1]==target).float(),good)),
        proof_action_accuracy=float(mean(torch.stack(matches,1).float(),mask)))
    if return_parts:return loss,metrics,dict(one_step_dynamics=dynamics,subgoal_prediction=goal_loss,proof_action_imitation=planning)
    return loss,metrics


def transition_objective(model,batch,prepared):
    require_state_only(batch);b=len(batch['task']);c=model.cfg
    with torch.no_grad(),torch.autocast(device_type=batch['task'].device.type,enabled=False):
        observed,memory,previous,codes=model.encode_sequence(prepared,batch['prefix_codes']);before=observed[:,-1].argmax(-1)
        future_features=model.observation(model.embed(batch['successors'])).flatten(0,1)
        nm,nf,_=model.advance(memory.repeat_interleave(c.actions,0),previous.repeat_interleave(c.actions,0),
            codes.repeat_interleave(c.actions,0),future_features,batch['successors'].flatten(0,1))
        features=F.normalize(nf+model.memory_projection(nm),dim=-1)
        definitions=F.normalize(prepared['definitions'],dim=-1).repeat_interleave(c.actions,0)
        target=torch.einsum('bd,bkd->bk',features,definitions).argmax(-1).reshape(b,c.actions)
    probabilities=model.rows(prepared,before[:,None])[:,0]
    loss=-probabilities.gather(-1,target[:,:,None]).squeeze(-1).clamp_min(1e-8).log().mean()
    return loss,dict(counterfactual_dynamics=float(loss.detach()),counterfactual_loss=float(loss.detach()),
        counterfactual_accuracy=float((probabilities.argmax(-1)==target).float().mean()))
