"""ACDF-inspired categorical forward + multi-step inverse learning.

Every forward row is grounded in recorded outcomes. Planning/action imitation
cannot rewrite the world table to invent a successful edge. Auxiliary neural
inverse/reconstruction heads are discarded by the runtime interpreter.
"""
import math
import torch
from torch.nn import functional as F
from model import hard


def entropy(probabilities):
    return -(probabilities.clamp_min(1e-8).log()*probabilities).sum(-1)

def ce(logits,labels,mask=None):
    value=F.cross_entropy(logits.flatten(0,-2),labels.flatten(),reduction='none').reshape(labels.shape)
    return value.mean() if mask is None else (value*mask).sum()/mask.sum().clamp_min(1)

def accuracy(logits,labels,mask=None):
    value=(logits.argmax(-1)==labels).float()
    return value.mean() if mask is None else (value*mask).sum()/mask.sum().clamp_min(1)

def objective(model,data):
    c=model.cfg;x=data['features'];previous=data['previous'];batch=len(x)
    index=torch.arange(batch,device=x.device);start=data['start'];end=data['end']
    observation_logits=model.observe(x);observations=hard(observation_logits)
    memory_logits,memory_hidden,_=model.memory_sequence(observations,previous)
    memories=hard(memory_logits)
    contexts=model.context_sequence(x,observations,previous,data['task'])
    context=contexts[index,start];definitions=model.definitions(context)
    # Detached labels are categorical codes of ACTUAL successor observations,
    # not predictions of the world model. Stable global definitions avoid a
    # fresh, arbitrary renumbering at every program revision.
    target_o=observation_logits.detach().argmax(-1)
    target_m=memory_logits.detach().argmax(-1)
    sm,ob=memories[index,start],observations[index,start]
    losses=[];memory_correct=[];obs_correct=[];one_memory=[];one_obs=[]
    endpoint_m=None;endpoint_o=None
    for k in range(10):
        t=(start+k).clamp_max(x.shape[1]-2)
        valid=(start+k<end)&data['valid'][index,t]
        action=data['actions'][index,t]
        next_m,next_o=model.transition(context,sm,ob,action,definitions)
        tm=target_m[index,t+1];to=target_o[index,t+1]
        losses.append(ce(next_m,tm,valid)+ce(next_o,to,valid))
        memory_correct.append(accuracy(next_m,tm,valid));obs_correct.append(accuracy(next_o,to,valid))
        # Independent teacher-forced accuracy: same row decoder, known current
        # codes. This is logged separately from autonomous ten-step recurrence.
        known_m,known_o=model.transition(context,memories[index,t],observations[index,t],action,definitions)
        losses.append(.5*(ce(known_m,tm,valid)+ce(known_o,to,valid)))
        one_memory.append(accuracy(known_m,tm,valid));one_obs.append(accuracy(known_o,to,valid))
        sm,ob=hard(next_m),hard(next_o)
        endpoint_m,endpoint_o=sm,ob
    forward=torch.stack(losses).mean()
    gm,go=model.goal_logits(context,data['task'])
    # Final demonstrated state AFTER STOP; no fixed semantic STOP predicate.
    goal=ce(gm,target_m[index,end])+ce(go,target_o[index,end])
    offsets=torch.randint(1,11,(batch,),device=x.device)
    future=torch.minimum(start+offsets,end);distance=(future-start).clamp(1,10)
    inverse_input=torch.cat([observations[index,start],observations[index,future],F.one_hot(distance-1,10).float()],-1)
    inverse_logits=model.inverse(inverse_input)
    inverse=ce(inverse_logits,data['actions'][index,start],data['valid'][index,start])

    bx=data['branch_features'];bprevious=data['branch_previous'];bsize=len(bx)
    bi=torch.arange(bsize,device=x.device);bt=data['branch_t'];bs=data['branch_start']
    bo_logits=model.observe(bx);bo=hard(bo_logits)
    bm_logits,_,_=model.memory_sequence(bo,bprevious);bm=hard(bm_logits)
    bc=model.context_sequence(bx,bo,bprevious,data['branch_task'])[bi,bs]
    successors=data['successors'];so_logits=model.observe(successors);so=hard(so_logits)
    a=torch.arange(c.actions,device=x.device)[None].expand(bsize,-1)
    # Causal memory at t plus the actual observed result of each possible a.
    m0=bm[bi,bt][:,None].expand(-1,c.actions,-1)
    o0=bo[bi,bt][:,None].expand(-1,c.actions,-1)
    successor_m,_=model.memory_step(m0,o0,a)
    pred_m,pred_o=model.transition(bc,m0,o0,a)
    targets_m=successor_m.detach().argmax(-1);targets_o=so_logits.detach().argmax(-1)
    branch=ce(pred_m,targets_m)+ce(pred_o,targets_o)
    persistence_logits=model.persistence_logits(bc,m0,o0,a)
    persistence_targets=(o0.detach().argmax(-1)==targets_o)[...,None].float()
    persistence_loss=F.binary_cross_entropy_with_logits(persistence_logits,persistence_targets)
    branch_inv=model.inverse(torch.cat([o0,so,
        F.one_hot(torch.zeros_like(a),10).float()],-1))
    # Identical observed effects are legitimate equivalent inverse actions.
    difference=(successors[:,:,None].float()-successors[:,None,:].float()).square().mean((-1,-2))
    equivalent=(difference<1e-10).float()
    branch_inverse=-(branch_inv.log_softmax(-1)*equivalent/equivalent.sum(-1,keepdim=True)).sum(-1).mean()
    actual_mask=torch.arange(x.shape[1],device=x.device)[None]<=end[:,None]
    obs_prob=observation_logits.softmax(-1)[actual_mask]
    mem_prob=memory_logits.softmax(-1)[actual_mask]
    # Preserve visual evidence, while forward/inverse losses make the compact
    # representation control-relevant. No object/location labels are provided.
    reconstruction=F.mse_loss(model.reconstruct(observations[actual_mask]),x[actual_mask].float().flatten(-2))
    memory_reconstruction=F.mse_loss(model.memory_reconstruct(memories[actual_mask]),memory_hidden.detach()[actual_mask])
    usage=(math.log(c.observations)-entropy(obs_prob.mean(0))+
           math.log(c.memories)-entropy(mem_prob.mean(0)))
    confidence=.5*(entropy(obs_prob).mean()+entropy(mem_prob).mean())
    # Renewal learning only predicts whether old rule predictions disagree with
    # actual encoded outcomes. It never chooses a motor action.
    ages=(bt-bs).float()[:,None].expand(-1,c.actions);pm,po=pred_m.detach(),pred_o.detach()
    renew_target=((pm.argmax(-1)!=targets_m)|(po.argmax(-1)!=targets_o)).float()
    renewal_logits=model.renewal_logits(hard(pm),so,hard(po),ages)
    renewal=F.binary_cross_entropy_with_logits(renewal_logits,renew_target)
    operator_logits=model.operator_class(torch.arange(c.actions,device=x.device))
    operator_loss=F.cross_entropy(operator_logits,data['operator_targets'])
    total=forward+branch+goal+.5*inverse+.5*branch_inverse+.5*reconstruction+memory_reconstruction+.25*usage+.02*confidence+.1*renewal+.5*persistence_loss+operator_loss
    metrics=dict(loss=total.detach(),forward_loss=forward.detach(),branch_loss=branch.detach(),goal_loss=goal.detach(),
        inverse_loss=inverse.detach(),branch_inverse_loss=branch_inverse.detach(),reconstruction=reconstruction.detach(),
        usage_loss=usage.detach(),renewal_loss=renewal.detach(),
        single_step_observation_accuracy=torch.stack(one_obs).mean(),single_step_memory_accuracy=torch.stack(one_memory).mean(),
        rollout_observation_accuracy=torch.stack(obs_correct).mean(),rollout_memory_accuracy=torch.stack(memory_correct).mean(),
        branch_observation_accuracy=accuracy(pred_o,targets_o),branch_memory_accuracy=accuracy(pred_m,targets_m),
        goal_observation_accuracy=accuracy(go,target_o[index,end]),goal_memory_accuracy=accuracy(gm,target_m[index,end]),
        inverse_accuracy=accuracy(inverse_logits,data['actions'][index,start],data['valid'][index,start]),
        observation_codes_used=obs_prob.argmax(-1).unique().numel(),memory_codes_used=mem_prob.argmax(-1).unique().numel(),
        observation_perplexity=entropy(obs_prob.mean(0)).exp(),memory_perplexity=entropy(mem_prob.mean(0)).exp(),
        branch_observation_persistence_baseline=(bo_logits[bi,bt].argmax(-1)[:,None]==targets_o).float().mean(),
        branch_changed_observation_accuracy=accuracy(pred_o,targets_o,bo_logits[bi,bt].argmax(-1)[:,None]!=targets_o),
        hard_observation_perplexity=entropy(observations.detach()[actual_mask].mean(0)).exp(),
        hard_memory_perplexity=entropy(memories.detach()[actual_mask].mean(0)).exp(),
        persistence_loss=persistence_loss.detach(),persistence_observation_accuracy=((persistence_logits[...,0]>0)==persistence_targets[...,0].bool()).float().mean(),
        recorded_action0_observation_accuracy=accuracy(pred_o[:,0],targets_o[:,0]),
        predicted_action0_observation_identity=(pred_o[:,0].argmax(-1)==o0[:,0].argmax(-1)).float().mean(),
        operator_class_loss=operator_loss.detach(),operator_class_accuracy=(operator_logits.argmax(-1)==data['operator_targets']).float().mean(),
        learned_global_preserve_actions=(operator_logits.argmax(-1)==1).sum())
    return total,metrics
