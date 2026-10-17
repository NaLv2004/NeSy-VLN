"""Learn anonymous definitions, rule transitions and future symbolic subgoals."""
import torch
import torch.nn.functional as F
from .subgoal_model import straight_through
from .subgoal_data import require_state_only


def mean(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)


def transition(program,state,actions):
    selectors=straight_through(program['parents'])
    marginal=torch.einsum('bfpg,bgk->bfpk',selectors,state)
    premises=marginal[:,:,0,:,None]*marginal[:,:,1,None,:]
    same=selectors[:,:,0]*selectors[:,:,1]
    premises=premises-torch.einsum('bfg,bgu,bgv->bfuv',same,state,state)+torch.diag_embed(torch.einsum('bfg,bgk->bfk',same,state))
    b,f=state.shape[:2];table=program['transitions'][torch.arange(b,device=state.device)[:,None],torch.arange(f,device=state.device)[None],actions[:,None]]
    return torch.einsum('bfuv,bfuvk->bfk',premises,table)


def all_transitions(program,state):
    selectors=straight_through(program['parents'])
    marginal=torch.einsum('bfpg,bgk->bfpk',selectors,state)
    premises=marginal[:,:,0,:,None]*marginal[:,:,1,None,:]
    same=selectors[:,:,0]*selectors[:,:,1]
    premises=premises-torch.einsum('bfg,bgu,bgv->bfuv',same,state,state)+torch.diag_embed(torch.einsum('bfg,bgk->bfk',same,state))
    return torch.einsum('bfuv,bfauvk->bafk',premises,program['transitions'])


def posterior(model,program,codes,actions):return model.training_targets(program,codes,actions)


def future_prediction(model,batch):
    require_state_only(batch);start=int(batch['start'][0]);width=min(start+1,model.cfg.history)
    if not torch.all(batch['start']==start):raise ValueError('Mixed boundaries')
    program=model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)
    state=straight_through(model.ground(program,batch['targets'][:,0]));probabilities=[]
    for i in range(10):
        result=transition(program,state,batch['actions'][:,i]);probabilities.append(result);state=straight_through(result)
    return program,torch.stack(probabilities,1)


def objective(model,batch):
    program,predicted=future_prediction(model,batch);c=model.cfg;b=len(batch['task'])
    observed=posterior(model,program,batch['targets'],batch['actions']);hard=straight_through(observed)
    target=hard[:,1:].detach();valid=batch['action_valid'].float();states=batch['state_valid'].float()
    dynamics=-(target*predicted.clamp_min(1e-8).log()).sum(-1).mean(-1)
    representation=-(observed[:,1:]*predicted.detach().clamp_min(1e-8).log()).sum(-1).mean(-1)
    reconstruction=model.reconstruct(hard.flatten(-2)).reshape(b,11,c.patches,c.vocabulary)
    reconstruction=F.cross_entropy(reconstruction.flatten(0,2),batch['targets'].flatten(),reduction='none').reshape(b,11,-1).mean(-1)
    inverse=model.inverse(torch.cat([hard[:,:-1,:c.observed_factors].flatten(-2),hard[:,1:,:c.observed_factors].flatten(-2)],-1))
    inverse=F.cross_entropy(inverse.flatten(0,1),batch['actions'].flatten(),reduction='none').reshape(b,10)
    # Infer the demonstrated action through the generated transition rules.
    # There is no action-scoring neural module in this path. Gradients also
    # discourage different observed successors from collapsing into one code.
    initial=straight_through(model.ground(program,batch['targets'][:,0]))
    rule_inverse=[]
    for i in range(10):
        state=initial if i==0 else straight_through(predicted[:,i-1])
        alternatives=all_transitions(program,state).clamp_min(1e-8).log()
        logits=torch.einsum('bafk,bfk->ba',alternatives,hard[:,i+1])
        rule_inverse.append(F.cross_entropy(logits,batch['actions'][:,i],reduction='none'))
    rule_inverse=torch.stack(rule_inverse,1)
    desired=hard[torch.arange(b,device=hard.device),batch['goal_index']].detach()
    goal_logp=program['subgoal_logits'].log_softmax(-1)
    goal_ce=-(desired*goal_logp).sum(-1).mean()
    endpoint=predicted[torch.arange(b,device=hard.device),batch['goal_index']-1]
    path_loss=-(desired*endpoint.clamp_min(1e-8).log()).sum(-1).mean()
    usage=(hard*states[:,:,None,None]).sum((0,1))/states.sum().clamp_min(1)
    balance=(usage*(usage.clamp_min(1e-8)*c.states).log()).sum(-1).mean()
    entropy=mean(-(observed*observed.clamp_min(1e-8).log()).sum(-1).mean(-1),states)
    losses=dict(dynamics=mean(dynamics,valid),posterior_alignment=mean(representation,valid),reconstruction=mean(reconstruction,states),
        inverse=mean(inverse,valid),rule_inverse=mean(rule_inverse,valid),subgoal_prediction=goal_ce,teacher_path_goal=path_loss,balance=balance,posterior_entropy=entropy)
    weights=dict(dynamics=1.,posterior_alignment=.1,reconstruction=.5,inverse=.2,rule_inverse=1.,subgoal_prediction=1.,teacher_path_goal=.5,balance=.05,posterior_entropy=.01)
    total=sum(weights[k]*v for k,v in losses.items());metrics={k:float(v.detach()) for k,v in losses.items()}
    goal_ids=program['subgoal_logits'].argmax(-1);desired_ids=desired.argmax(-1)
    metrics.update(loss=float(total.detach()),subgoal_factor_accuracy=float((goal_ids==desired_ids).float().mean()),
        subgoal_exact_accuracy=float((goal_ids==desired_ids).all(-1).float().mean()),
        teacher_path_exact_goal=float((endpoint.argmax(-1)==desired_ids).all(-1).float().mean()),
        transition_factor_accuracy=float(mean((predicted.argmax(-1)==target.argmax(-1)).float().mean(-1),valid)))
    return total,metrics


def transition_objective(model,batch):
    require_state_only(batch);b=len(batch['task']);c=model.cfg;device=batch['task'].device
    lengths=torch.full((b,),batch['history'].shape[1],device=device,dtype=torch.long)
    program=model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=batch['start'])
    state=straight_through(model.ground(program,batch['prefix_codes'][:,0]));memory=program['context_raw']
    for i in range(batch['offset']):
        state=straight_through(transition(program,state,batch['prefix_actions'][:,i]))
        x=model.context_read(model.embed(batch['prefix_codes'][:,i+1]))+model.action_embed(batch['prefix_actions'][:,i])
        _,h=model.history(x[:,None],memory[None]);memory=h[0]
    predicted=torch.stack([transition(program,state,torch.full((b,),a,device=device)) for a in range(c.actions)],1)
    action=torch.arange(c.actions,device=device)[None].expand(b,-1)
    x=model.context_read(model.embed(batch['successors']))+model.action_embed(action)
    _,h=model.history(x.reshape(b*c.actions,1,-1),memory.repeat_interleave(c.actions,0)[None])
    defs={'definitions':program['definitions'].repeat_interleave(c.actions,0)}
    observed=model.ground_features(defs,batch['successors'].flatten(0,1),h[0]).reshape(b,c.actions,c.factors,c.states)
    hard=straight_through(observed)
    dynamics=-(hard.detach()*predicted.clamp_min(1e-8).log()).sum(-1).mean()
    representation=-(observed*predicted.detach().clamp_min(1e-8).log()).sum(-1).mean()
    reconstruction=model.reconstruct(hard.flatten(-2)).reshape(b,c.actions,c.patches,c.vocabulary)
    reconstruction=F.cross_entropy(reconstruction.flatten(0,2),batch['successors'].flatten())
    rule_logits=torch.einsum('bafk,bjfk->bja',predicted.clamp_min(1e-8).log(),hard)
    rule_inverse=F.cross_entropy(rule_logits.flatten(0,1),action.flatten())
    loss=dynamics+.1*representation+.5*reconstruction+rule_inverse
    return loss,dict(counterfactual_dynamics=float(dynamics.detach()),counterfactual_reconstruction=float(reconstruction.detach()),
        counterfactual_factor_accuracy=float((predicted.argmax(-1)==hard.argmax(-1)).float().mean()),
        counterfactual_rule_inverse=float(rule_inverse.detach()),counterfactual_rule_inverse_accuracy=float((rule_logits.argmax(-1)==action).float().mean()),
        counterfactual_loss=float(loss.detach()))
