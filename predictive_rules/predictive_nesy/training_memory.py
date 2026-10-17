"""Ten-step probability propagation, learned memory, and full-table supervision."""
import torch
import torch.nn.functional as F
from .model import straight_through
from .model_memory import transition_all,risk_all,scalar,scores
from .training import weighted_mean


def bce(probability,label):
    p=probability.float().clamp(1e-6,1-1e-6)
    return -label*p.log()-(1-label)*torch.log1p(-p)


def objective(model,batch):
    if not torch.all(batch['start']==batch['start'][0]):raise ValueError('Mixed boundaries')
    start=int(batch['start'][0]);c=model.cfg;width=min(start+1,c.history)
    program=model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)
    observed=model.training_targets(program,batch['targets'],batch['actions'])
    hard=straight_through(observed)
    valid=batch['valid'].float();supervised=valid*batch['supervised']
    b,t,f,k=observed.shape;of=c.observed_factors
    labels=batch['teacher_ids']
    # Only existing observation factors are anchored. Predictive memory is free
    # to distinguish states that share exactly the same old observation IDs.
    grounding=-observed[:,:,:of].clamp_min(1e-8).log().gather(-1,labels[...,None]).squeeze(-1).mean(-1)
    reconstruction=model.reconstruct(hard.flatten(-2)).reshape(b,t,c.patches,c.vocabulary)
    reconstruction=F.cross_entropy(reconstruction.flatten(0,2),batch['targets'].flatten(),reduction='none').reshape(b,t,-1).mean(-1)
    target=torch.cat([F.one_hot(labels,k).float(),hard[:,:,of:].detach()],2)
    # The only propagated path starts here. Future recognized states never
    # replace this belief. Teacher actions specify which rule to apply in SFT.
    state=hard[:,0]
    dynamics=[];dmask=[];actor=[];goal=[];value=[];risk=[];bellman=[];accuracy=[];symbol=[]
    goal_observed=scalar(program,hard,'goals',model.pairs)
    value_observed=scalar(program,hard,'values',model.pairs)
    # Discounted demonstrated return includes the same step costs as planning.
    target_value=(batch['value']-.02*(1-batch['value'])/(1-.97)).clamp(0,1)
    for step in range(t):
        q=scores(program,state,model.pairs)
        act=batch['actions'][:,step]
        actor.append(F.cross_entropy(q/.1,act,reduction='none'))
        accuracy.append((q.argmax(-1)==act).float())
        current_goal=scalar(program,state,'goals',model.pairs)
        current_value=scalar(program,state,'values',model.pairs)
        goal.append(bce(current_goal,batch['goal'][:,step].float()))
        value.append((current_value-target_value[:,step]).square())
        selected=q.gather(1,act[:,None]).squeeze(1)
        bellman.append((current_value-selected).square())
        known=batch['risks'][:,step]
        risk.append(bce(risk_all(program,state),known.clamp_min(0)))
        symbol.append((state[:,:of].argmax(-1)==labels[:,step]).float().mean(-1))
        if step<t-1:
            result=transition_all(program,state)[torch.arange(b,device=state.device),act]
            dynamics.append(-(target[:,step+1]*result.clamp_min(1e-8).log()).sum(-1).mean(-1))
            dmask.append(valid[:,step]*valid[:,step+1]*(act!=0))
            state=result  # Categorical probabilities; no argmax and no encoder.
    losses=dict(grounding=weighted_mean(grounding,valid),reconstruction=weighted_mean(reconstruction,valid),
        dynamics=weighted_mean(torch.stack(dynamics,1),torch.stack(dmask,1)),
        planning=weighted_mean(torch.stack(actor,1),supervised),
        goal=weighted_mean(torch.stack(goal,1)+bce(goal_observed,batch['goal'].float()),supervised),
        value=weighted_mean(torch.stack(value,1)+(value_observed-target_value).square(),supervised),
        bellman=weighted_mean(torch.stack(bellman,1),supervised),
        risk=weighted_mean(torch.stack(risk,1),(batch['risks']>=0)*valid[...,None]))
    before=hard[:,:-1,:of].flatten(-2);after=hard[:,1:,:of].flatten(-2)
    inv=model.inverse(torch.cat([before,after],-1))
    inverse=F.cross_entropy(inv.flatten(0,1),batch['actions'][:,:-1].flatten(),reduction='none').reshape(b,t-1)
    losses['inverse']=weighted_mean(inverse,valid[:,:-1]*valid[:,1:])
    # Occupancy is computed on actual categorical choices (straight-through),
    # so uniform logits selecting one category do not pass the usage objective.
    use=(hard[:,:,of:]*valid[:,:,None,None]).sum((0,1))/valid.sum().clamp_min(1)
    losses['memory_usage']=(use*(use.clamp_min(1e-8)*k).log()).sum(-1).mean()
    confidence=-(observed[:,:,of:]*observed[:,:,of:].clamp_min(1e-8).log()).sum(-1).mean(-1)
    losses['memory_entropy']=weighted_mean(confidence,valid)
    weights=dict(grounding=.5,reconstruction=.2,dynamics=1.,planning=1.,goal=1.,value=2.,bellman=.2,
                 risk=.5,inverse=.1,memory_usage=.05,memory_entropy=.01)
    total=sum(weights[key]*loss for key,loss in losses.items())
    metrics={key:float(value.detach()) for key,value in losses.items()}
    metrics.update(loss=float(total.detach()),open_loop_action_accuracy=float(weighted_mean(torch.stack(accuracy,1),supervised)),
        open_loop_observed_state_accuracy=float(weighted_mean(torch.stack(symbol,1),valid)),
        grounding_accuracy=float(weighted_mean((observed[:,:,:of].argmax(-1)==labels).float().mean(-1),valid)),
        t10_samples=int(valid[:,-1].sum()),state_accuracy_t10=float(weighted_mean(symbol[-1],valid[:,-1])) if valid[:,-1].sum()>0 else None)
    return total,metrics
