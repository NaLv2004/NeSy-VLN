"""Rectangular robust reachability using ONLY exported categorical tables.

For each integer state/action, the environment's row is allowed to be any of
the fitted ensemble rows. Maximize the minimum goal probability over those
choices. No reward network, semantic label, threshold or physical map.
"""
from dataclasses import dataclass,field
import hashlib
import numpy as np

@dataclass
class Program:
    next_memory:np.ndarray
    outcome_probability:np.ndarray  # [members, joint states, actions, O]
    goal:int
    observations:int=64
    _levels:list=field(default_factory=list,repr=False)
    _failures:list=field(default_factory=list,repr=False)
    _policies:list=field(default_factory=list,repr=False)
    _adversaries:list=field(default_factory=list,repr=False)
    _probability:np.ndarray=field(default=None,repr=False)
    def validate(self):
        t=self.next_memory;p=self.outcome_probability
        assert np.issubdtype(t.dtype,np.integer) and t.ndim==2 and t.shape[1]==17
        assert p.ndim==4 and p.shape[1:]==(*t.shape,self.observations) and len(p)>0
        assert t.min()>=0 and t.max()<len(t)//self.observations
        assert np.isfinite(p).all() and p.min()>=0 and np.allclose(p.sum(-1),1,atol=2e-6)
        assert 0<=self.goal<len(t)
        if self._probability is None:
            self._probability=np.asarray(p,dtype=np.float64)
            self._probability=self._probability/self._probability.sum(-1,keepdims=True)
        return self
    def digest(self):
        return hashlib.sha256(self.next_memory.tobytes()+self.outcome_probability.tobytes()+str(self.goal).encode()).hexdigest()
    def extend(self,horizon):
        self.validate();n,actions=self.next_memory.shape
        if not self._levels:
            self._levels=[np.zeros(n)];self._failures=[np.ones(n)]
            self._policies=[np.zeros(n,np.int16)];self._adversaries=[np.zeros((n,actions),np.int16)]
        gm,go=divmod(self.goal,self.observations);p=self._probability
        stop=np.where(self.next_memory[None,:,0]==gm,p[:,:,0,go],0.)
        non_goal=np.arange(self.observations)!=go
        # Use basic slicing before boolean indexing to preserve [K,S,O].
        stop_failure=np.where(self.next_memory[None,:,0]==gm,p[:,:,0,:][...,non_goal].sum(-1),1.)
        for _ in range(len(self._levels),horizon+1):
            prev=self._levels[-1].reshape(-1,self.observations)
            prev_f=self._failures[-1].reshape(-1,self.observations)
            q=(p*prev[self.next_memory][None]).sum(-1)
            f=(p*prev_f[self.next_memory][None]).sum(-1)
            q[:,:,0]=stop;f[:,:,0]=stop_failure
            qmin=q.min(0);fmax=f.max(0)
            adversary=np.where(qmin<=.5,q.argmin(0),f.argmax(0))
            rows=np.arange(n)[:,None];cols=np.arange(actions)[None]
            worst_q=q[adversary,rows,cols];worst_f=f[adversary,rows,cols]
            action=np.where(worst_q.max(-1)<=.5,worst_q.argmax(-1),worst_f.argmin(-1))
            self._levels.append(np.clip(worst_q[np.arange(n),action],0,1))
            self._failures.append(np.clip(worst_f[np.arange(n),action],0,1))
            self._policies.append(action.astype(np.int16));self._adversaries.append(adversary.astype(np.int16))
    def row(self,state,action,horizon):
        member=int(self._adversaries[horizon][state,action])
        return member,self._probability[member,state,action]

def plan(program,memory,observed,horizon=32):
    assert 1<=horizon<=32
    program.extend(horizon);initial=int(memory)*program.observations+int(observed)
    value=float(program._levels[horizon][initial])
    result=dict(found=value>0,goal=program.goal,memory=int(memory),observed=int(observed),horizon=horizon,
        model_goal_probability=value,model_failure_probability=float(program._failures[horizon][initial]),
        states=[initial],actions=[],branch_probability=[],adversary_members=[],program_sha256=program.digest(),
        derivation='Finite-horizon max-action/min-ensemble-row categorical reachability')
    if not result['found']:return result
    current=initial
    for remaining in range(horizon,0,-1):
        action=int(program._policies[remaining][current]);nm=int(program.next_memory[current,action])
        member,p=program.row(current,action,remaining)
        future=nm*program.observations+np.arange(program.observations)
        if action==0:
            chosen=program.goal%program.observations
            assert nm==program.goal//program.observations and p[chosen]>0
            continuation=(future==program.goal).astype(float);failure=(future!=program.goal).astype(float)
        else:
            continuation=program._levels[remaining-1][future];failure=program._failures[remaining-1][future]
            mass=p*continuation;chosen=int(mass.argmax());assert mass[chosen]>0
        if remaining==horizon:
            result.update(next_memory=nm,predicted_observation=int(p.argmax()),
                root_outcome_probabilities=p.tolist(),root_continuation_probabilities=continuation.tolist(),
                root_continuation_failures=failure.tolist(),root_worst_member=member)
        result['actions'].append(action);result['adversary_members'].append(member)
        result['branch_probability'].append(float(p[chosen]));current=int(future[chosen]);result['states'].append(current)
        if action==0:break
    audit(program,result);return result

def audit(program,proof):
    assert proof['found'] and proof['program_sha256']==program.digest()
    current=proof['memory']*program.observations+proof['observed'];remaining=proof['horizon']
    assert proof['states'][0]==current and proof['actions'][-1]==0
    for i,action in enumerate(proof['actions']):
        assert action==int(program._policies[remaining][current])
        member,p=program.row(current,action,remaining);assert member==proof['adversary_members'][i]
        nm,ob=divmod(proof['states'][i+1],program.observations)
        assert nm==int(program.next_memory[current,action]) and p[ob]>0
        if action==0:assert i==len(proof['actions'])-1 and proof['states'][i+1]==program.goal
        current=proof['states'][i+1];remaining-=1
    assert np.isclose(np.dot(proof['root_outcome_probabilities'],proof['root_continuation_probabilities']),proof['model_goal_probability'],rtol=2e-6,atol=1e-12)
    assert np.isclose(np.dot(proof['root_outcome_probabilities'],proof['root_continuation_failures']),proof['model_failure_probability'],rtol=2e-12,atol=1e-300)
    assert np.isclose(proof['model_goal_probability']+proof['model_failure_probability'],1,atol=1e-12)
    return True
