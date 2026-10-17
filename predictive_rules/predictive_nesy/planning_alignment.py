"""Train generated tables against the multi-step planner's own alternatives.

No observations or neural networks are called while expanding symbolic states.
This module implements the deployed reward and probability-table semantics.
"""
import torch
from .model import straight_through
from .model_memory import scalar


def branch_joint(program,state):
    selectors=straight_through(program['parents'])
    parents=torch.einsum('bfpg,bmgk->bmfpk',selectors,state)
    result=parents[:,:,:,0,:,None]*parents[:,:,:,1,None,:]
    same=selectors[:,:,0]*selectors[:,:,1]
    result=result-torch.einsum('bfg,bmgu,bmgv->bmfuv',same,state,state)
    return result+torch.diag_embed(torch.einsum('bfg,bmgk->bmfk',same,state))


def branch_outcomes(program,state):
    weights=branch_joint(program,state)
    after=torch.einsum('bmfuv,bfauvk->bmafk',weights,program['transitions'])
    risk=torch.einsum('bmfuv,bfauv->bma',weights,program['risks'])/state.shape[-2]
    return after,risk


@torch.no_grad()
def beam_candidates(program,initial,pairs,horizon=10,beam_width=8):
    b,f,k=initial.shape;a=program['transitions'].shape[2];device=initial.device
    state=initial[:,None];returns=torch.zeros(b,1,device=device);terminal=torch.zeros(b,1,dtype=torch.bool,device=device)
    paths=torch.zeros(b,1,0,dtype=torch.long,device=device)
    for depth in range(horizon):
        after,risk=branch_outcomes(program,state)
        goal=scalar(program,state,'goals',pairs)
        actions=torch.arange(a,device=device)[None,None].expand(b,state.shape[1],-1)
        stopped=terminal[:,:,None]|(actions==0)
        after=torch.where(stopped[:,:,:,None,None],state[:,:,None],after)
        reward=torch.where(actions==0,2*goal[:,:,None]-1,-.02-.2*risk)
        reward=torch.where(terminal[:,:,None],0.,reward)
        new_return=returns[:,:,None]+.97**depth*reward
        score=new_return+torch.where(stopped,0.,.97**(depth+1)*scalar(program,after,'values',pairs))
        score=score.masked_fill(terminal[:,:,None]&(actions!=0),-torch.inf)
        kept=score.flatten(1).argsort(dim=-1,descending=True,stable=True)[:,:beam_width]
        parent=kept//a;act=kept%a;index=torch.arange(b,device=device)[:,None]
        state=after[index,parent,act];returns=new_return[index,parent,act];terminal=stopped[index,parent,act]
        paths=torch.cat([paths[index,parent],act[:,:,None]],-1)
    return paths,score.flatten(1).gather(1,kept)


def sequence_scores(program,initial,sequences,pairs):
    b,n,h=sequences.shape;state=initial[:,None].expand(-1,n,-1,-1)
    total=torch.zeros(b,n,device=state.device);stopped=torch.zeros(b,n,dtype=torch.bool,device=state.device)
    index=torch.arange(b,device=state.device)[:,None];branch=torch.arange(n,device=state.device)[None]
    for depth in range(h):
        act=sequences[:,:,depth];weights=branch_joint(program,state)
        factor=torch.arange(state.shape[-2],device=state.device)[None,None]
        chosen_table=program['transitions'][index[:,:,None],factor,act[:,:,None]]
        after=torch.einsum('bmfuv,bmfuvk->bmfk',weights,chosen_table)
        chosen_risk=program['risks'][index[:,:,None],factor,act[:,:,None]]
        risk=torch.einsum('bmfuv,bmfuv->bm',weights,chosen_risk)/state.shape[-2]
        goal=scalar(program,state,'goals',pairs)
        reward=torch.where(act==0,2*goal-1,-.02-.2*risk.detach())
        total=total+.97**depth*torch.where(stopped,0.,reward)
        stopped=stopped|(act==0)
        state=torch.where(stopped[:,:,None,None],state,after)
    return total+torch.where(stopped,0.,.97**h*scalar(program,state,'values',pairs))


def loss(model,batch,beam_width=8):
    if not torch.all(batch['start']==batch['start'][0]):raise ValueError('Mixed revision boundaries')
    start=int(batch['start'][0]);width=min(start+1,model.cfg.history)
    program=model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)
    initial=straight_through(model.ground(program,batch['targets'][:,0]))
    horizon=min(10,32-start)
    candidates,beam_scores=beam_candidates(program,initial,model.pairs,horizon=horizon,beam_width=beam_width)
    # Use the strongest generated alternative with a different first action.
    # Selection is discrete; gradients are through the tables scoring the two
    # complete paths, with collision probabilities trained from measurements.
    wrong=candidates[:,:,0]!=batch['actions'][:,0,None]
    wrong_score=beam_scores.masked_fill(~wrong,-torch.inf)
    competitor=wrong_score.argmax(1)
    negative=candidates[torch.arange(len(initial),device=initial.device),competitor][:,None]
    teacher=batch['actions'][:,:horizon,None].transpose(1,2)
    scores=sequence_scores(program,initial,torch.cat([teacher,negative],1),model.pairs)
    valid=wrong.any(1)&batch['supervised'][:,0].bool()
    margin=.05+scores[:,1]-scores[:,0]
    result=(margin.relu()*valid).sum()/valid.sum().clamp_min(1)
    return result,dict(planning_alignment_loss=float(result.detach()),
        beam_first_action_match=float((candidates[:,0,0]==batch['actions'][:,0]).float().mean()),
        competing_sequences=int(valid.sum()),teacher_sequence_score=float(scores[:,0].mean().detach()),
        alternative_sequence_score=float(scores[:,1].mean().detach()))
