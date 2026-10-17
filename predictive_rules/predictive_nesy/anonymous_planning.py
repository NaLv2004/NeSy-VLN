"""Differentiable scoring of complete anonymous-table plans."""
import torch
from .anonymous_model import straight_through,transition_all,parent_joint,scalar,reward_all


@torch.no_grad()
def beam_candidates(program,initial,pairs,horizon=10,beam_width=32,*,terminal_horizon=False):
    b=initial.shape[0];a=program['transitions'].shape[2];device=initial.device
    state=initial[:,None];returns=torch.zeros(b,1,device=device);terminal=torch.zeros(b,1,dtype=torch.bool,device=device)
    paths=torch.zeros(b,1,0,dtype=torch.long,device=device)
    for depth in range(horizon):
        after=transition_all(program,state);rewards=reward_all(program,state,pairs)
        actions=torch.arange(a,device=device)[None,None].expand(b,state.shape[1],-1)
        stopped=terminal[:,:,None]|(actions==0)
        after=torch.where(stopped[:,:,:,None,None],state[:,:,None],after)
        total=returns[:,:,None]+.97**depth*torch.where(terminal[:,:,None],0.,rewards)
        final_budget=terminal_horizon and depth+1==horizon
        score=total if final_budget else total+torch.where(stopped,0.,.97**(depth+1)*scalar(program,after,'values',pairs))
        score=score.masked_fill(terminal[:,:,None]&(actions!=0),-torch.inf)
        kept=score.flatten(1).argsort(dim=-1,descending=True,stable=True)[:,:beam_width]
        parent=kept//a;act=kept%a;index=torch.arange(b,device=device)[:,None]
        state=after[index,parent,act];returns=total[index,parent,act];terminal=stopped[index,parent,act]
        paths=torch.cat([paths[index,parent],act[:,:,None]],-1)
    return paths,score.flatten(1).gather(1,kept)


def sequence_scores(program,initial,sequences,pairs,*,terminal_horizon=False):
    b,n,h=sequences.shape;state=initial[:,None].expand(-1,n,-1,-1)
    total=torch.zeros(b,n,device=state.device);stopped=torch.zeros(b,n,dtype=torch.bool,device=state.device)
    index=torch.arange(b,device=state.device)[:,None];factor=torch.arange(state.shape[-2],device=state.device)[None,None]
    for depth in range(h):
        action=sequences[:,:,depth];premises=parent_joint(program,state)
        table=program['transitions'][index[:,:,None],factor,action[:,:,None]]
        after=torch.einsum('bmfuv,bmfuvk->bmfk',premises,table)
        r=reward_all(program,state,pairs,fixed_table=True).gather(-1,action[...,None]).squeeze(-1)
        total=total+.97**depth*torch.where(stopped,0.,r)
        stopped=stopped|(action==0);state=torch.where(stopped[:,:,None,None],state,after)
    return total if terminal_horizon else total+torch.where(stopped,0.,.97**h*scalar(program,state,'values',pairs))


def loss(model,batch,beam_width=32):
    start=int(batch['start'][0]);width=min(start+1,model.cfg.history);horizon=min(10,32-start)
    if not torch.all(batch['start']==start):raise ValueError('Mixed boundaries')
    program=model.generate(batch['task'],batch['history'][:,:width],batch['previous_actions'][:,:width],batch['history_length'],start=start)
    initial=straight_through(model.ground(program,batch['targets'][:,0]))
    terminal_horizon=start+horizon==32
    paths,values=beam_candidates(program,initial,model.pairs,horizon,beam_width,terminal_horizon=terminal_horizon)
    wrong=paths[:,:,0]!=batch['actions'][:,0,None]
    competitor=values.masked_fill(~wrong,-torch.inf).argmax(1)
    negative=paths[torch.arange(len(initial),device=initial.device),competitor][:,None]
    teacher=batch['actions'][:,:horizon,None].transpose(1,2)
    score=sequence_scores(program,initial,torch.cat([teacher,negative],1),model.pairs,terminal_horizon=terminal_horizon)
    complete=(batch['supervised'][:,:horizon].bool()|~batch['valid'][:,:horizon].bool()).all(1)
    valid=wrong.any(1)&complete&batch['supervised'][:,0].bool()
    result=((.05+score[:,1]-score[:,0]).relu()*valid).sum()/valid.sum().clamp_min(1)
    return result,dict(planning_alignment_loss=float(result.detach()),competing_sequences=int(valid.sum()),
        beam_first_action_match=float((paths[:,0,0]==batch['actions'][:,0]).float().mean()))
