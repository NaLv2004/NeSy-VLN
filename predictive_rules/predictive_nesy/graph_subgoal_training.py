"""Learn complete goal-directed derivations, not a reward or action table."""
import math
import torch
import torch.nn.functional as F
from .graph_subgoal_model import straight_through
from .subgoal_data import require_state_only


def mean(x,mask):return (x*mask).sum()/mask.sum().clamp_min(1)


def transition(program,state,action):
    table=program['transitions'][torch.arange(len(state),device=state.device),:,action]
    # The executable rule is a categorical row lookup. AMP matrix products can
    # round two distinct probabilities into a tie and change the successor ID.
    exact=table[torch.arange(len(state),device=state.device),state.argmax(-1)]
    if not torch.is_grad_enabled():return exact
    surrogate=torch.einsum('bs,bsk->bk',state,table)
    return exact.detach()+(surrogate-surrogate.detach()).float()


def reachability(program,goal,horizon):
    """Derived support for complete goal proofs; no learned value function.

    STOP closes a proof. All other actions recurse through a generated rule.
    The last movement may reach the subgoal at the block horizon.
    """
    table=program['transitions'];support=goal;levels=[None]
    direct=torch.einsum('bsak,bk->bsa',table,goal)
    for depth in range(1,horizon+1):
        if depth==1:q=direct
        else:
            next_support=torch.einsum('bsak,bk->bsa',table,support)
            q=torch.cat([direct[:,:,:1],next_support[:,:,1:]],-1)
        levels.append(q);support=q.amax(-1)
    return levels


def proof_log_support(program,goal,horizon):
    """Same discrete transitions and proof likelihood as the NumPy solver.

    Straight-through derivatives are a training surrogate only. The forward
    pass follows the argmax successor of each rule, exactly as execution does.
    Finite support for a failed proof permits gradients. Its bound is below
    every possible valid length-H proof: max row probability is at least 1/K.
    It is not a learned reward, value, semantic classifier or executable fallback.
    """
    probabilities=program['transitions'];rules=straight_through(probabilities)
    factor=horizon+1;scale=10000
    confidence=_DiscreteLogConfidence.apply(probabilities.max(-1).values,scale,factor)
    unreachable=math.ceil((horizon+1)*math.log(probabilities.shape[-1])+1)*scale*factor
    terminal=-unreachable*(1-goal);support=terminal;levels=[None]
    direct=rule_support(rules,terminal)+confidence
    for depth in range(1,horizon+1):
        if depth==1:q=direct
        else:
            through_rule=rule_support(rules,support)+confidence
            q=torch.cat([direct[:,:,:1],through_rule[:,:,1:]],-1)
        levels.append(q);support=q.amax(-1)
    return levels


def rule_support(rules,support):
    """Exact lookup forward; linear-rule surrogate backward only.

    A mixed-precision matrix product must never round the integer proof scores.
    The explicitly parenthesized zero leaves the gathered value unchanged.
    """
    ids=rules.argmax(-1)
    exact=support.gather(1,ids.flatten(1)).reshape_as(ids)
    if not torch.is_grad_enabled():return exact
    surrogate=torch.einsum('bsak,bk->bsa',rules,support)
    return exact.detach()+(surrogate-surrogate.detach()).float()


def state_support(state,support):
    ids=state.argmax(-1)
    exact=support[torch.arange(len(state),device=state.device),ids]
    if not torch.is_grad_enabled():return exact
    surrogate=torch.einsum('bs,bsa->ba',state,support)
    return exact.detach()+(surrogate-surrogate.detach()).float()


class _DiscreteLogConfidence(torch.autograd.Function):
    @staticmethod
    def forward(ctx,probability,scale,factor):
        ctx.save_for_backward(probability);ctx.multiplier=scale*factor
        return torch.round(probability.double().log()*scale).float()*factor-1
    @staticmethod
    def backward(ctx,gradient):
        probability,=ctx.saved_tensors
        return gradient*ctx.multiplier/probability,None,None


def objective(model,batch):
    require_state_only(batch);b=len(batch['task']);device=batch['task'].device;start=int(batch['start'][0]);c=model.cfg;horizon=min(10,32-start)
    assert torch.all(batch['start']==start)
    width=min(start+1,11);p=model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)
    observed=model.training_targets(p,batch['targets'],batch['actions']).squeeze(2);hard=straight_through(observed)
    state=straight_through(model.ground(p,batch['targets'][:,0]).squeeze(1));before=[];prediction=[]
    for i in range(10):
        before.append(state);prob=transition(p,state,batch['actions'][:,i]);prediction.append(prob);state=straight_through(prob)
    predicted=torch.stack(prediction,1);current=torch.stack(before,1)
    valid=batch['action_valid'].float();state_mask=batch['state_valid'].float();imitation=valid*batch['supervised'].float()
    future=hard[:,1:];dynamics=mean(-(future.detach()*predicted.clamp_min(1e-8).log()).sum(-1),valid)
    alignment=mean(-(observed[:,1:]*predicted.detach().clamp_min(1e-8).log()).sum(-1),valid)
    reconstruction=model.reconstruct_codes(p,hard)
    reconstruction=mean(F.cross_entropy(reconstruction.flatten(0,2),batch['targets'].flatten(),reduction='none').reshape(b,11,-1).mean(-1),state_mask)
    desired=hard[torch.arange(b,device=device),batch['goal_index']]
    goal=straight_through(p['subgoal_logits'].squeeze(1).softmax(-1))
    good_window=((batch['supervised']|~batch['action_valid']).all(-1)).float()
    goal_loss=mean(-(desired.detach()*p['subgoal_logits'].squeeze(1).log_softmax(-1)).sum(-1),good_window)
    endpoint=predicted[torch.arange(b,device=device),batch['goal_index']-1]
    endpoint_loss=mean(-(desired.detach()*endpoint.clamp_min(1e-8).log()).sum(-1),good_window)
    # The training action distribution is derived from the generated rules and
    # generated goal. There is no separate neural actor or return prediction.
    levels=proof_log_support(p,goal,horizon);plan_losses=[];correct=[]
    for i in range(horizon):
        support=state_support(current[:,i],levels[horizon-i])
        logits=support/(10000*(horizon+1)*0.5)
        plan_losses.append(F.cross_entropy(logits,batch['actions'][:,i],reduction='none'))
        correct.append(logits.argmax(-1)==batch['actions'][:,i])
    plan_losses=torch.stack(plan_losses,1);plan_mask=imitation[:,:horizon]*good_window[:,None]
    planning=mean(plan_losses,plan_mask)
    all_next=torch.einsum('bts,bsak->btak',current,p['transitions'])
    inverse_logits=torch.einsum('btak,btk->bta',all_next.clamp_min(1e-8).log(),future)
    inverse=mean(F.cross_entropy(inverse_logits.flatten(0,1),batch['actions'].flatten(),reduction='none').reshape(b,10),imitation)
    usage=(hard*state_mask[:,:,None]).sum((0,1))/state_mask.sum().clamp_min(1)
    balance=(usage*(usage.clamp_min(1e-8)*c.states).log()).sum()
    entropy=mean(-(observed*observed.clamp_min(1e-8).log()).sum(-1),state_mask)
    losses=dict(dynamics=dynamics,posterior_alignment=alignment,reconstruction=reconstruction,subgoal_prediction=goal_loss,
        teacher_path_goal=endpoint_loss,proof_action_imitation=planning,rule_inverse=inverse,balance=balance,posterior_entropy=entropy)
    weights=dict(dynamics=1.,posterior_alignment=.1,reconstruction=.5,subgoal_prediction=1.,teacher_path_goal=.5,
        proof_action_imitation=2.,rule_inverse=.3,balance=.2,posterior_entropy=.01)
    loss=sum(weights[k]*v for k,v in losses.items());metrics={k:float(v.detach()) for k,v in losses.items()}
    metrics.update(loss=float(loss.detach()),subgoal_exact_accuracy=float(mean((goal.argmax(-1)==desired.argmax(-1)).float(),good_window)),
        teacher_path_exact_goal=float(mean((endpoint.argmax(-1)==desired.argmax(-1)).float(),good_window)),
        transition_accuracy=float(mean((predicted.argmax(-1)==future.argmax(-1)).float(),valid)),
        proof_action_accuracy=float(mean(torch.stack(correct,1).float(),plan_mask)))
    return loss,metrics


def transition_objective(model,batch):
    require_state_only(batch);b=len(batch['task']);c=model.cfg;device=batch['task'].device
    lengths=torch.full((b,),batch['history'].shape[1],device=device,dtype=torch.long)
    p=model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=batch['start'])
    state=straight_through(model.ground(p,batch['prefix_codes'][:,0]).squeeze(1));memory=p['context_raw']
    for i in range(batch['offset']):
        state=straight_through(transition(p,state,batch['prefix_actions'][:,i]))
        x=model.context_read(model.embed(batch['prefix_codes'][:,i+1]))+model.action_embed(batch['prefix_actions'][:,i])
        _,hidden=model.history(x[:,None],memory[None]);memory=hidden[0]
    predicted=torch.einsum('bs,bsak->bak',state,p['transitions'])
    actions=torch.arange(c.actions,device=device)[None].expand(b,-1)
    x=model.context_read(model.embed(batch['successors']))+model.action_embed(actions)
    _,hidden=model.history(x.reshape(b*c.actions,1,-1),memory.repeat_interleave(c.actions,0)[None])
    definitions=dict(definitions=p['definitions'].repeat_interleave(c.actions,0))
    observed=model.ground_features(definitions,batch['successors'].flatten(0,1),hidden[0]).reshape(b,c.actions,c.states);hard=straight_through(observed)
    dynamics=-(hard.detach()*predicted.clamp_min(1e-8).log()).sum(-1).mean()
    alignment=-(observed*predicted.detach().clamp_min(1e-8).log()).sum(-1).mean()
    reconstruction=F.cross_entropy(model.reconstruct_codes(p,hard).flatten(0,2),batch['successors'].flatten())
    # Action information inside memory can make inverse prediction trivial.
    # Keep this weak auxiliary term; the complete goal proof is the main action supervision.
    inverse_logits=torch.einsum('bak,bjk->bja',predicted.clamp_min(1e-8).log(),hard)
    inverse=F.cross_entropy(inverse_logits.flatten(0,1),actions.flatten())
    loss=dynamics+.1*alignment+.5*reconstruction+.1*inverse
    return loss,dict(counterfactual_loss=float(loss.detach()),counterfactual_dynamics=float(dynamics.detach()),
        counterfactual_accuracy=float((predicted.argmax(-1)==hard.argmax(-1)).float().mean()),
        counterfactual_inverse=float(inverse.detach()))
