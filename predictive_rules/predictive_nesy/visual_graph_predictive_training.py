"""Visual state labels and teacher-forced connections for goal-directed rules."""
import torch
from torch.nn import functional as F
from .graph_subgoal_model import straight_through
from .graph_subgoal_training import mean,transition,proof_log_support,state_support
from .subgoal_data import require_state_only


def objective(model,batch):
    require_state_only(batch);b=len(batch['task']);device=batch['task'].device;start=int(batch['start'][0]);c=model.cfg;h=min(10,32-start)
    assert torch.all(batch['start']==start)
    width=min(start+1,11)
    p=model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)
    observed=model.training_targets(p,batch['targets'],batch['actions']).squeeze(2);hard=straight_through(observed)
    valid=batch['action_valid'].float();state_mask=batch['state_valid'].float();imitation=valid*batch['supervised'].float()
    current=hard[:,:-1];future=hard[:,1:]
    all_next=torch.einsum('bts,bsak->btak',current,p['transitions'])
    teacher_next=all_next.gather(2,batch['actions'][:,:,None,None].expand(-1,-1,1,c.states)).squeeze(2).float()
    one_step=mean(-(future.detach()*teacher_next.clamp_min(1e-8).log()).sum(-1),valid)
    alignment=mean(-(observed[:,1:]*teacher_next.detach().clamp_min(1e-8).log()).sum(-1),valid)
    state=hard[:,0];rollout=[]
    for t in range(10):
        predicted=transition(p,state,batch['actions'][:,t]);rollout.append(predicted);state=straight_through(predicted)
    predicted=torch.stack(rollout,1)
    open_loop=mean(-(future.detach()*predicted.clamp_min(1e-8).log()).sum(-1),valid)
    reconstructed=model.reconstruct_codes(p,hard)
    reconstruction=mean(F.cross_entropy(reconstructed.flatten(0,2),batch['targets'].flatten(),reduction='none').reshape(b,11,-1).mean(-1),state_mask)
    desired=hard[torch.arange(b,device=device),batch['goal_index']]
    goal=straight_through(p['subgoal_logits'].squeeze(1).softmax(-1));good=((batch['supervised']|~batch['action_valid']).all(-1)).float()
    goal_loss=mean(-(desired.detach()*p['subgoal_logits'].squeeze(1).log_softmax(-1)).sum(-1),good)
    endpoint=predicted[torch.arange(b,device=device),batch['goal_index']-1]
    endpoint_loss=mean(-(desired.detach()*endpoint.clamp_min(1e-8).log()).sum(-1),good)
    levels=proof_log_support(p,goal,h);planning_losses=[];correct=[]
    for t in range(h):
        # Teacher forcing is supervision of a generated world model, not a
        # runtime observation correction. No future input enters the generator.
        scores=state_support(current[:,t].detach(),levels[h-t])/(10000*(h+1)*.5)
        planning_losses.append(F.cross_entropy(scores,batch['actions'][:,t],reduction='none'))
        correct.append(scores.argmax(-1)==batch['actions'][:,t])
    plan_mask=imitation[:,:h]*good[:,None]
    planning=mean(torch.stack(planning_losses,1),plan_mask)
    inverse_logits=torch.einsum('btak,btk->bta',all_next.clamp_min(1e-8).log(),future)
    inverse=mean(F.cross_entropy(inverse_logits.flatten(0,1),batch['actions'].flatten(),reduction='none').reshape(b,10),imitation)
    usage=(hard*state_mask[:,:,None]).sum((0,1))/state_mask.sum().clamp_min(1)
    balance=(usage*(usage.clamp_min(1e-8)*c.states).log()).sum()
    entropy=mean(-(observed*observed.clamp_min(1e-8).log()).sum(-1),state_mask)
    losses=dict(one_step_dynamics=one_step,dynamics=open_loop,posterior_alignment=alignment,reconstruction=reconstruction,
        subgoal_prediction=goal_loss,teacher_path_goal=endpoint_loss,proof_action_imitation=planning,
        rule_inverse=inverse,balance=balance,posterior_entropy=entropy)
    weights=dict(one_step_dynamics=2.,dynamics=.5,posterior_alignment=1.,reconstruction=.5,subgoal_prediction=1.,
        teacher_path_goal=.5,proof_action_imitation=1.,rule_inverse=.3,balance=.2,posterior_entropy=.01)
    loss=sum(weights[k]*v for k,v in losses.items());metrics={k:float(v.detach()) for k,v in losses.items()}
    metrics.update(loss=float(loss.detach()),subgoal_exact_accuracy=float(mean((goal.argmax(-1)==desired.argmax(-1)).float(),good)),
        teacher_path_exact_goal=float(mean((endpoint.argmax(-1)==desired.argmax(-1)).float(),good)),
        one_step_accuracy=float(mean((teacher_next.argmax(-1)==future.argmax(-1)).float(),valid)),
        transition_accuracy=float(mean((predicted.argmax(-1)==future.argmax(-1)).float(),valid)),
        proof_action_accuracy=float(mean(torch.stack(correct,1).float(),plan_mask)))
    return loss,metrics


def transition_objective(model,batch):
    require_state_only(batch);b=len(batch['task']);c=model.cfg;device=batch['task'].device
    lengths=torch.full((b,),batch['history'].shape[1],device=device,dtype=torch.long)
    p=model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=batch['start'])
    # Actual observations below are targets/premises of supervised transition
    # examples only. The executable interpreter never receives these images.
    observed_before=model.ground(p,batch['prefix_codes'][:,-1]).squeeze(1)
    before=straight_through(observed_before)
    predicted=torch.einsum('bs,bsak->bak',before,p['transitions']).float()
    definitions={'definitions':p['definitions'].repeat_interleave(c.actions,0)}
    observed=model.ground_features(definitions,batch['successors'].flatten(0,1)).reshape(b,c.actions,c.states)
    hard=straight_through(observed)
    dynamics=-(hard.detach()*predicted.clamp_min(1e-8).log()).sum(-1).mean()
    alignment=-(observed*predicted.detach().clamp_min(1e-8).log()).sum(-1).mean()
    state=straight_through(model.ground(p,batch['prefix_codes'][:,0]).squeeze(1))
    for t in range(batch['offset']):state=straight_through(transition(p,state,batch['prefix_actions'][:,t]))
    propagated=torch.einsum('bs,bsak->bak',state,p['transitions']).float()
    open_loop=-(hard.detach()*propagated.clamp_min(1e-8).log()).sum(-1).mean()
    reconstruction=F.cross_entropy(model.reconstruct_codes(p,hard).flatten(0,2),batch['successors'].flatten())
    loss=dynamics+.5*open_loop+1.*alignment+.5*reconstruction
    return loss,dict(counterfactual_loss=float(loss.detach()),counterfactual_dynamics=float(dynamics.detach()),
        counterfactual_open_loop=float(open_loop.detach()),counterfactual_accuracy=float((predicted.argmax(-1)==hard.argmax(-1)).float().mean()))
