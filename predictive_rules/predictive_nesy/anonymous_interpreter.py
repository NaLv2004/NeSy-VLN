"""Executable anonymous state tables and generic scalar-return search. NumPy only."""
from dataclasses import dataclass
from itertools import combinations
import hashlib
import numpy as np


FIELDS=('parents','transitions','rewards','values','weights','definitions')


def joint(program,state):
    left=state[...,program.parents[:,0],:];right=state[...,program.parents[:,1],:]
    result=left[...,:,None]*right[...,None,:]
    for f,(i,j) in enumerate(program.parents):
        if i==j:result[...,f,:,:]=left[...,f,:,None]*np.eye(state.shape[-1],dtype=state.dtype)
    return result


def successors(program,state):
    return np.einsum('...fuv,fauvk->...afk',joint(program,state),program.transitions,optimize=True)


def scalar(program,state,name):
    pairs=list(combinations(range(state.shape[-2]),2))
    left=state[...,[i for i,j in pairs],:];right=state[...,[j for i,j in pairs],:]
    return (np.einsum('...pu,...pv,puv->...p',left,right,getattr(program,name),optimize=True)*program.weights).sum(-1)


def as_belief(program,state):
    state=np.asarray(state);f,_,k,_,_=program.transitions.shape
    if state.shape==(f,):
        if state.dtype.kind not in 'iu':raise ValueError('Expected categorical IDs')
        state=np.eye(k,dtype=np.float32)[state]
    if state.shape!=(f,k) or np.any(state<0) or not np.allclose(state.sum(-1),1,atol=2e-5):raise ValueError('Invalid categorical belief')
    return state.astype(np.float32,copy=True)


@dataclass(frozen=True)
class Program:
    parents: np.ndarray
    transitions: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    definitions: np.ndarray
    start: int
    interval: int=10

    def validate(self):
        f,a,k,k2,k3=self.transitions.shape;p=f*(f-1)//2
        assert k==k2==k3 and self.parents.shape==(f,2)
        assert self.parents.dtype.kind in 'iu' and ((self.parents>=0)&(self.parents<f)).all()
        assert self.rewards.shape==(p,a,k,k) and self.values.shape==(p,k,k)
        assert self.weights.shape==(p,) and self.definitions.shape[:2]==(f,k)
        for name in FIELDS:assert np.isfinite(getattr(self,name)).all()
        assert (self.transitions>=0).all() and np.allclose(self.transitions.sum(-1),1,atol=1e-5)
        assert (self.weights>=0).all() and np.isclose(self.weights.sum(),1,atol=1e-5)
        assert (np.abs(self.rewards)<=1).all() and (np.abs(self.values)<=1).all()
        assert self.start>=0 and self.start%self.interval==0
        return self

    def digest(self):
        h=hashlib.sha256()
        for name in FIELDS:h.update(np.ascontiguousarray(getattr(self,name)).tobytes())
        h.update(str((self.start,self.interval)).encode());return h.hexdigest()

    def save(self,path):np.savez_compressed(path,**self.__dict__)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!=set(FIELDS)|{'start','interval'}:raise ValueError('Unexpected executable fields')
            values={name:data[name] for name in data.files}
        values['start']=int(values['start']);values['interval']=int(values['interval'])
        return cls(**values).validate()


def export_program(generated,index=0):
    if set(generated)!=set(FIELDS)|{'start','context_raw'}:raise ValueError('Unexpected generated parameters')
    arrays={name:generated[name][index].detach().float().cpu().numpy().copy() for name in FIELDS if name!='parents'}
    arrays['parents']=generated['parents'][index].detach().argmax(-1).cpu().numpy().astype(np.int64)
    return Program(**arrays,start=generated['start']).validate()


def reward_all(program,state):
    pairs=list(combinations(range(state.shape[-2]),2))
    left=state[...,[i for i,j in pairs],:];right=state[...,[j for i,j in pairs],:]
    rows=np.einsum('...pu,...pv,pauv->...pa',left,right,program.rewards,optimize=True)
    return (rows*program.weights[:,None]).sum(-2)


def plan(program,initial,horizon=10,beam_width=32,gamma=.97,*,terminal_horizon=False):
    program.validate()
    if not 1<=horizon<=program.interval:raise ValueError('Plan crosses revision boundary')
    beliefs=as_belief(program,initial)[None];paths=np.zeros((1,0),np.int64)
    returns=np.zeros(1);terminal=np.zeros(1,bool);states=beliefs[:,None].copy();expanded=0
    a=program.transitions.shape[1]
    for depth in range(horizon):
        after=successors(program,beliefs);rewards=reward_all(program,beliefs)
        parent=np.repeat(np.arange(len(beliefs)),a);actions=np.tile(np.arange(a),len(beliefs))
        alive=~terminal[parent];valid=alive|(actions==0)
        parent,actions,alive=parent[valid],actions[valid],alive[valid]
        stopped=(actions==0)|~alive
        next_states=np.where(stopped[:,None,None],beliefs[parent],after[parent,actions])
        total=returns[parent]+gamma**depth*np.where(alive,rewards[parent,actions],0)
        final_budget=terminal_horizon and depth+1==horizon
        score=total if final_budget else total+np.where(stopped,0,gamma**(depth+1)*scalar(program,next_states,'values'))
        kept=np.argsort(-score,kind='stable')[:beam_width]
        beliefs=next_states[kept];returns=total[kept];terminal=stopped[kept]
        paths=np.concatenate([paths[parent[kept]],actions[kept,None]],-1)
        states=np.concatenate([states[parent[kept]],beliefs[:,None]],1);expanded+=len(parent)
    selected=paths[0].tolist()
    if 0 in selected:selected=selected[:selected.index(0)+1]
    return dict(actions=selected,beliefs=states[0,:len(selected)+1].tolist(),states=states[0,:len(selected)+1].argmax(-1).tolist(),
        score=float(score[kept[0]]),expanded=expanded,program_sha256=program.digest(),planning_horizon=horizon,beam_width=beam_width,
        terminal_horizon=terminal_horizon,
        state_semantics='anonymous finite categorical variables; scalar task return is not a state or predicate')


class Controller:
    def __init__(self):
        self.interval=10;self.step=0;self.queue=[];self.finished=False
        self.current=None;self.program=None

    def revise(self,program,state,*,horizon=10,beam_width=32):
        if self.finished or self.step%10 or self.queue or program.start!=self.step:raise ValueError('Illegal revision')
        self.program=program.validate();self.current=as_belief(program,state)
        result=plan(program,self.current,horizon=horizon,beam_width=beam_width,terminal_horizon=self.step+horizon==32)
        self.queue=result['actions'].copy();return result

    def next_action(self):
        if self.finished or not self.queue:raise ValueError('Revision or episode end required')
        action=self.queue.pop(0);before=self.current.copy()
        if action:self.current=successors(self.program,self.current)[action]
        else:self.finished=True;self.queue.clear()
        proof=dict(step=self.step,program_start=self.program.start,action=action,before=before.argmax(-1).tolist(),
            after=self.current.argmax(-1).tolist(),before_belief=before.tolist(),after_belief=self.current.tolist(),
            program_sha256=self.program.digest())
        self.step+=1;return action,proof
