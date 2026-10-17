"""Neural-free planning by marginalization over finite symbolic states.

Beliefs are probabilities over named finite categories, not continuous visual
embeddings. Only exported conditional tables operate on them. Cross-factor
independence is an explicit approximation, not a claim of a full joint belief.
"""
from itertools import combinations
import numpy as np
from .interpreter import Program


def joint(program,belief):
    p=belief[...,program.parents[:,0],:]
    q=belief[...,program.parents[:,1],:]
    result=p[..., :,None]*q[...,None,:]
    for f,(i,j) in enumerate(program.parents):
        if i==j:
            result[...,f,:,:]=p[...,f,:,None]*np.eye(p.shape[-1],dtype=p.dtype)
    return result


def successors(program,belief):
    # Fixed contraction order avoids the large naive five-dimensional broadcast.
    weights=joint(program,belief)
    return np.einsum('...fuv,fauvk->...afk',weights,program.transitions,optimize=True)


def risk_all(program,belief):
    return np.einsum('...fuv,fauv->...a',joint(program,belief),program.risks,optimize=True)/belief.shape[-2]


def scalar(program,belief,name):
    pairs=list(combinations(range(belief.shape[-2]),2))
    p=belief[...,[i for i,j in pairs],:]
    q=belief[...,[j for i,j in pairs],:]
    result=np.einsum('...pu,...pv,puv->...p',p,q,getattr(program,name),optimize=True)
    return (result*program.weights).sum(-1)


def as_belief(program,state):
    state=np.asarray(state)
    f,a,k,_,_=program.transitions.shape
    if state.shape==(f,):
        if state.dtype.kind not in 'iu':raise TypeError('Categorical IDs must be integers')
        state=np.eye(k,dtype=np.float32)[state]
    if state.shape!=(f,k) or np.any(state<0) or not np.allclose(state.sum(-1),1,atol=2e-5):
        raise ValueError('Invalid categorical belief')
    return state.astype(np.float32,copy=True)


def plan(program,initial,horizon=10,beam_width=32,gamma=.97,step_cost=.02,collision_cost=.2):
    program.validate()
    if not 1<=horizon<=program.interval:raise ValueError('Plan crosses a revision boundary')
    beliefs=as_belief(program,initial)[None]
    paths=np.zeros((1,0),dtype=np.int64);returns=np.zeros(1)
    terminal=np.zeros(1,dtype=bool);states=beliefs[:,None].copy();expanded=0
    a=program.transitions.shape[1]
    for depth in range(horizon):
        next_beliefs=successors(program,beliefs)
        risks=risk_all(program,beliefs)
        before_goals=scalar(program,beliefs,'goals')
        parent=np.repeat(np.arange(len(beliefs)),a);actions=np.tile(np.arange(a),len(beliefs))
        alive=~terminal[parent];valid=alive|(actions==0)
        parent,actions,alive=parent[valid],actions[valid],alive[valid]
        after=next_beliefs[parent,actions]
        stopped=(actions==0)|~alive
        after=np.where(stopped[:,None,None],beliefs[parent],after)
        rewards=-step_cost-collision_cost*risks[parent,actions]
        rewards=np.where(actions==0,2*before_goals[parent]-1,rewards)
        rewards=np.where(alive,rewards,0)
        total=returns[parent]+gamma**depth*rewards
        score=total+np.where(stopped,0,gamma**(depth+1)*scalar(program,after,'values'))
        kept=np.argsort(-score,kind='stable')[:beam_width]
        beliefs=after[kept];returns=total[kept];terminal=stopped[kept]
        paths=np.concatenate([paths[parent[kept]],actions[kept,None]],-1)
        states=np.concatenate([states[parent[kept]],beliefs[:,None]],1)
        expanded+=len(parent)
    selected=paths[0].tolist()
    if 0 in selected:selected=selected[:selected.index(0)+1]
    return dict(actions=selected,beliefs=states[0,:len(selected)+1].tolist(),
        states=states[0,:len(selected)+1].argmax(-1).tolist(),score=float(score[kept[0]]),expanded=expanded,
        program_sha256=program.digest(),planning_horizon=horizon,beam_width=beam_width,
        state_semantics='factorized categorical probability, correlated repeated premises')


class BeliefController:
    def __init__(self,interval=10):
        self.interval=interval;self.step=0;self.queue=[];self.finished=False
        self.program=None;self.current=None;self.program_hash=None

    def revise(self,program,state,*,horizon=None,beam_width=32):
        if self.finished or self.step%self.interval or self.queue:
            raise ValueError('No observation correction between revisions')
        if program.start!=self.step:raise ValueError('Wrong boundary program')
        self.program=program.validate();self.program_hash=program.digest()
        self.current=as_belief(program,state)
        result=plan(program,self.current,horizon=horizon or self.interval,beam_width=beam_width)
        self.queue=result['actions'].copy()
        return result

    def next_action(self):
        if self.finished or not self.queue:raise ValueError('Boundary revision or episode end required')
        action=self.queue.pop(0);before=self.current.copy()
        if action:
            self.current=successors(self.program,self.current)[action]
        else:
            self.finished=True;self.queue.clear()
        proof=dict(step=self.step,program_start=self.program.start,action=action,
            before=before.argmax(-1).tolist(),after=self.current.argmax(-1).tolist(),
            before_belief=before.tolist(),after_belief=self.current.tolist(),
            program_sha256=self.program_hash)
        self.step+=1
        return action,proof
