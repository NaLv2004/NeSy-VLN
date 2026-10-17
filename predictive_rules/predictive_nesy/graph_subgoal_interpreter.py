"""Complete finite-state goal reachability using only exported rule arrays."""
from dataclasses import dataclass
import hashlib
import numpy as np

FIELDS=('transitions','definitions','subgoal')


@dataclass(frozen=True)
class Program:
    transitions:np.ndarray
    definitions:np.ndarray
    subgoal:np.ndarray
    start:int
    interval:int=10

    def validate(self):
        k,a,k2=self.transitions.shape
        assert k==k2 and self.definitions.shape[0]==k and self.subgoal.shape==(1,)
        assert self.subgoal.dtype.kind in 'iu' and 0<=self.subgoal[0]<k
        assert np.isfinite(self.transitions).all() and (self.transitions>=0).all()
        assert np.allclose(self.transitions.sum(-1),1,atol=1e-5)
        assert self.interval==10 and self.start>=0 and self.start%10==0
        for array in (self.transitions,self.definitions,self.subgoal):array.flags.writeable=False
        return self

    def digest(self):
        h=hashlib.sha256()
        for key in FIELDS:h.update(np.ascontiguousarray(getattr(self,key)).tobytes())
        h.update(str((self.start,self.interval)).encode());return h.hexdigest()

    def save(self,path):np.savez_compressed(path,**self.__dict__)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!=set(FIELDS)|{'start','interval'}:raise ValueError('Unexpected executable field')
            d={key:data[key] for key in data.files}
        d['start']=int(d['start']);d['interval']=int(d['interval']);return cls(**d).validate()


def export_program(generated,index=0):
    if set(generated)!={'transitions','definitions','subgoal_logits','start','context_raw'}:raise ValueError('Unexpected model fields')
    return Program(transitions=generated['transitions'][index].detach().float().cpu().numpy().copy(),
        definitions=generated['definitions'][index].detach().float().cpu().numpy().copy(),
        subgoal=generated['subgoal_logits'][index].detach().argmax(-1).cpu().numpy().astype(np.int64),start=generated['start']).validate()


def successors(program,state):
    state=np.asarray(state,dtype=np.int64);one=state.ndim==1
    if one:state=state[None]
    if state.shape[1]!=1:raise ValueError('Expected one anonymous joint-state ID')
    rows=program.transitions[state[:,0]];after=rows.argmax(-1)[...,None];confidence=np.log(rows.max(-1).clip(1e-8))
    return (after[0],confidence[0]) if one else (after,confidence)


def plan(program,initial,horizon=10,beam_width=None,*,terminal_goal=None):
    program.validate();initial=np.asarray(initial,dtype=np.int64)
    if initial.shape!=(1,) or not 1<=horizon<=10:raise ValueError('Invalid planning request')
    table=program.transitions;next_id=table.argmax(-1);k,a=next_id.shape
    # Integer log-confidence units give deterministic agreement with the
    # differentiable training interpreter, including equivalent-path ties.
    # The primary criterion is rule confidence; equal values prefer fewer steps.
    confidence=np.rint(np.log(table.max(-1).astype(np.float64))*10000).astype(np.int64)*(horizon+1)-1
    impossible=-10**15;exact=np.full(k,impossible,dtype=np.int64);exact[int(program.subgoal[0])]=0;support=exact;policies=[None]
    stop_exact=exact.copy()
    if terminal_goal is not None:
        terminal_goal=np.asarray(terminal_goal,dtype=np.int64)
        if terminal_goal.shape!=(1,) or not 0<=terminal_goal[0]<k:raise ValueError('Invalid terminal goal')
        stop_exact[np.arange(k)!=terminal_goal[0]]=impossible
    for depth in range(1,horizon+1):
        derivation=confidence+support[next_id]
        derivation[:,0]=confidence[:,0]+stop_exact[next_id[:,0]]
        policy=derivation.argmax(-1);policies.append(policy);support=derivation[np.arange(k),policy]
    base=dict(subgoal=program.subgoal.tolist(),expanded=horizon*k*a,planning_horizon=horizon,
        exhaustive_finite_state_search=True,program_sha256=program.digest())
    if support[initial[0]]<impossible//2:return dict(**base,found=False,actions=[],states=[initial.tolist()],goal_distance=None)
    state=int(initial[0]);states=[[state]];actions=[]
    for remaining in range(horizon,0,-1):
        action=int(policies[remaining][state]);actions.append(action);state=int(next_id[state,action]);states.append([state])
        if action==0:break
    assert state==int(program.subgoal[0])
    return dict(**base,found=True,actions=actions,states=states,goal_distance=0)


class Controller:
    def __init__(self):self.step=0;self.queue=[];self.finished=False;self.program=None;self.current=None
    def revise(self,program,state,*,horizon=10,beam_width=None):
        if self.finished or self.step%10 or self.queue or program.start!=self.step:raise ValueError('Revision outside boundary')
        self.program=program.validate();self.current=np.asarray(state,dtype=np.int64).copy()
        result=plan(program,self.current,horizon);self.queue=result['actions'].copy();self.program_sha256=result['program_sha256'];return result
    def next_action(self):
        if self.finished or not self.queue:raise ValueError('No proven symbolic plan')
        action=self.queue.pop(0);before=self.current.copy()
        self.current=np.array([self.program.transitions[int(before[0]),action].argmax()],dtype=np.int64)
        if action==0:self.finished=True;self.queue.clear()
        proof=dict(step=self.step,program_start=self.program.start,action=action,before=before.tolist(),after=self.current.tolist(),
            program_sha256=self.program_sha256,subgoal=self.program.subgoal.tolist(),subgoal_reached=bool(np.array_equal(self.current,self.program.subgoal)))
        self.step+=1;return action,proof
