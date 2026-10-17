"""Open-loop anonymous state learning and externally scored task returns."""
import torch
import torch.nn.functional as F
from .anonymous_model import straight_through,transition_all,scalar,reward_all,scores


def mean(x,mask):return (x*mask).sum()/mask.sum().clamp_min(1)


def require_anonymous(batch):
    allowed={'task','start','history','previous_actions','history_length','targets','actions','valid','supervised',
        'teacher_ids','scalar_rewards','returns','episode_id','offset','prefix_codes','prefix_actions','successors','episode_ids'}
    extra=set(batch)-allowed
    if extra:raise ValueError('Unexpected supervision fields: '+str(sorted(extra)))


def objective(model,batch):
    require_anonymous(batch)
    if not torch.all(batch['start']==batch['start'][0]):raise ValueError('Mixed boundaries')
    start=int(batch['start'][0]);c=model.cfg;width=min(start+1,c.history)
    program=model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)
    observed=model.training_targets(program,batch['targets'],batch['actions']);hard=straight_through(observed)
    valid=batch['valid'].float();supervised=valid*batch['supervised'];b,t,f,k=observed.shape;of=c.observed_factors
    labels=batch['teacher_ids'];target=torch.cat([F.one_hot(labels,k).float(),hard[:,:,of:].detach()],2)
    grounding=-observed[:,:,:of].clamp_min(1e-8).log().gather(-1,labels[...,None]).squeeze(-1).mean(-1)
    reconstruction=model.reconstruct(hard.flatten(-2)).reshape(b,t,c.patches,c.vocabulary)
    reconstruction=F.cross_entropy(reconstruction.flatten(0,2),batch['targets'].flatten(),reduction='none').reshape(b,t,-1).mean(-1)
    state=hard[:,0];dynamics=[];dmask=[];actor=[];values=[];rewards=[];consistency=[];matches=[];symbols=[]
    observed_r=reward_all(program,hard,model.pairs).gather(-1,batch['actions'][...,None]).squeeze(-1)
    observed_v=scalar(program,hard,'values',model.pairs)
    for step in range(t):
        act=batch['actions'][:,step];q=scores(program,state,model.pairs)
        actor.append(F.cross_entropy(q/.1,act,reduction='none'));matches.append((q.argmax(-1)==act).float())
        reward=reward_all(program,state,model.pairs).gather(1,act[:,None]).squeeze(1)
        value=scalar(program,state,'values',model.pairs)
        rewards.append((reward-batch['scalar_rewards'][:,step]).square())
        values.append((value-batch['returns'][:,step]).square())
        consistency.append((value-q.gather(1,act[:,None]).squeeze(1)).square())
        symbols.append((state[:,:of].argmax(-1)==labels[:,step]).float().mean(-1))
        if step<t-1:
            result=transition_all(program,state)[torch.arange(b,device=state.device),act]
            dynamics.append(-(target[:,step+1]*result.clamp_min(1e-8).log()).sum(-1).mean(-1))
            dmask.append(valid[:,step]*valid[:,step+1]*(act!=0));state=result
    losses=dict(grounding=mean(grounding,valid),reconstruction=mean(reconstruction,valid),
        dynamics=mean(torch.stack(dynamics,1),torch.stack(dmask,1)),
        action_ranking=mean(torch.stack(actor,1),supervised),
        scalar_reward=mean(torch.stack(rewards,1)+(observed_r-batch['scalar_rewards']).square(),supervised),
        scalar_return=mean(torch.stack(values,1)+(observed_v-batch['returns']).square(),supervised),
        bellman=mean(torch.stack(consistency,1),supervised))
    before=hard[:,:-1,:of].flatten(-2);after=hard[:,1:,:of].flatten(-2)
    inverse=model.inverse(torch.cat([before,after],-1))
    losses['inverse']=mean(F.cross_entropy(inverse.flatten(0,1),batch['actions'][:,:-1].flatten(),reduction='none').reshape(b,t-1),valid[:,:-1]*valid[:,1:])
    use=(hard[:,:,of:]*valid[:,:,None,None]).sum((0,1))/valid.sum().clamp_min(1)
    losses['memory_usage']=(use*(use.clamp_min(1e-8)*k).log()).sum(-1).mean()
    losses['memory_entropy']=mean(-(observed[:,:,of:]*observed[:,:,of:].clamp_min(1e-8).log()).sum(-1).mean(-1),valid)
    weights=dict(grounding=.5,reconstruction=.2,dynamics=1.,action_ranking=1.,scalar_reward=2.,scalar_return=2.,bellman=.2,inverse=.1,memory_usage=.05,memory_entropy=.01)
    total=sum(weights[key]*value for key,value in losses.items())
    metrics={key:float(value.detach()) for key,value in losses.items()}
    metrics.update(loss=float(total.detach()),open_loop_action_accuracy=float(mean(torch.stack(matches,1),supervised)),
        open_loop_observed_state_accuracy=float(mean(torch.stack(symbols,1),valid)),
        grounding_accuracy=float(mean((observed[:,:,:of].argmax(-1)==labels).float().mean(-1),valid)))
    return total,metrics


def outcome_predictions(model,batch):
    require_anonymous(batch);b=len(batch['task'])
    lengths=torch.full((b,),batch['history'].shape[1],device=batch['task'].device,dtype=torch.long)
    program=model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=batch['start'])
    state=straight_through(model.ground(program,batch['prefix_codes'][:,0]))
    for i in range(batch['offset']):
        state=transition_all(program,state)[torch.arange(b,device=state.device),batch['prefix_actions'][:,i]]
    after=transition_all(program,state)
    return program,state,torch.cat([state[:,None],after[:,1:]],1)


def outcome_objective(model,batch):
    b=len(batch['task']);c=model.cfg;program,state,predicted=outcome_predictions(model,batch)
    with torch.no_grad():
        memory=program['context_raw'].detach()
        for i in range(batch['offset']):
            z=model.context_read(model.embed(batch['prefix_codes'][:,i+1]))+model.action_embed(batch['prefix_actions'][:,i])
            _,h=model.history(z[:,None],memory[None]);memory=h[0]
        actions=torch.arange(c.actions,device=state.device)[None].expand(b,-1)
        z=model.context_read(model.embed(batch['successors']))+model.action_embed(actions)
        _,h=model.history(z.reshape(b*c.actions,1,-1),memory.repeat_interleave(c.actions,0)[None])
        next_memory=h[0].reshape(b,c.actions,-1);next_memory[:,0]=memory
    definitions={'definitions':program['definitions'].repeat_interleave(c.actions,0)}
    recognized=straight_through(model.ground_features(definitions,batch['successors'].flatten(0,1),next_memory.flatten(0,1))).reshape(b,c.actions,c.factors,c.states)
    current=recognized[:,0]
    transition=-(recognized.detach()*predicted.clamp_min(1e-8).log()).sum(-1).mean()
    reconstruction=model.reconstruct(recognized.flatten(-2)).reshape(b,c.actions,c.patches,c.vocabulary)
    reconstruction=F.cross_entropy(reconstruction.flatten(0,2),batch['successors'].flatten())
    target=batch['scalar_rewards'].float()
    def task_score_error(prediction):
        error=(prediction-target).square()
        return (error[:,0].mean()+error[:,1:].mean())/2
    calibration=(task_score_error(reward_all(program,state,model.pairs))+
        task_score_error(reward_all(program,current,model.pairs)))/2
    loss=transition+.2*reconstruction+4*calibration
    return loss,dict(outcome_loss=float(loss.detach()),outcome_transition=float(transition.detach()),
        outcome_reward_error=float(calibration.detach()),outcome_reconstruction=float(reconstruction.detach()))
