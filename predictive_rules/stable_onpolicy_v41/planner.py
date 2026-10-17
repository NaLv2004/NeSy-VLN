"""Finite-horizon probabilistic derivation over integer symbolic states.

No torch, images, actor, critic, neural search heuristic, or physical map.
The table contains all learned observation outcomes, not one modal successor.
Reachability is computed from that table and the unchanged generated goal.
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np


@dataclass
class Program:
    next_memory: np.ndarray
    outcome_probability: np.ndarray
    goal: int
    observations: int = 64
    _levels: list = field(default_factory=list,repr=False)
    _policies: list = field(default_factory=list,repr=False)
    _digest: str = field(default='',repr=False)

    def validate(self):
        t=self.next_memory;p=self.outcome_probability;n=len(t)
        assert np.issubdtype(t.dtype,np.integer) and t.ndim==2 and t.shape[1]==17
        assert t.min()>=0 and t.max()<n//self.observations
        assert p.shape==(*t.shape,self.observations) and np.isfinite(p).all()
        assert p.min()>=0 and np.allclose(p.sum(-1),1,atol=2e-6)
        assert 0<=self.goal<n
        return self

    def digest(self):
        if not self._digest:
            self._digest=hashlib.sha256(self.next_memory.tobytes()+self.outcome_probability.tobytes()+str(self.goal).encode()).hexdigest()
        return self._digest

    def extend(self,horizon):
        if not self._levels:
            self.validate()
            self._levels=[np.zeros(len(self.next_memory),dtype=np.float64)]
            self._policies=[np.zeros(len(self.next_memory),dtype=np.int16)]
        goal_m,goal_o=divmod(self.goal,self.observations)
        stop=np.where(self.next_memory[:,0]==goal_m,self.outcome_probability[:,0,goal_o],0.)
        while len(self._levels)<=horizon:
            previous=self._levels[-1].reshape(-1,self.observations)
            q=(self.outcome_probability*previous[self.next_memory]).sum(-1)
            # STOP terminates the physical episode and cannot precede another
            # action. Its only successful outcome is exactly the given goal.
            q[:,0]=stop
            action=q.argmax(-1)
            self._levels.append(q[np.arange(len(q)),action])
            self._policies.append(action.astype(np.int16))


def plan(program,memory,observed,horizon=32):
    assert 1<=horizon<=32
    program.extend(horizon)
    initial=int(memory)*program.observations+int(observed)
    value=float(program._levels[horizon][initial])
    result=dict(found=value>0,goal=program.goal,memory=int(memory),observed=int(observed),
                horizon=horizon,model_goal_probability=value,states=[initial],actions=[],
                branch_probability=[],program_sha256=program.digest(),
                derivation='Exact sum over learned discrete outcomes; max over rule-derived continuations')
    if not result['found']:return result
    current=initial
    # One successful branch is recorded for readable proof inspection, while
    # the selected action is based on ALL branches (not this witness alone).
    for remaining in range(horizon,0,-1):
        action=int(program._policies[remaining][current])
        nm=int(program.next_memory[current,action]);p=program.outcome_probability[current,action]
        future=nm*program.observations+np.arange(program.observations)
        if action==0:
            chosen=program.goal%program.observations
            assert nm==program.goal//program.observations and p[chosen]>0
        else:
            mass=p*program._levels[remaining-1][future]
            chosen=int(mass.argmax());assert mass[chosen]>0
        if remaining==horizon:
            result.update(next_memory=nm,predicted_observation=int(p.argmax()),
                          root_outcome_probabilities=p.tolist(),
                          root_continuation_probabilities=(future==program.goal).astype(float).tolist() if action==0
                          else program._levels[remaining-1][future].tolist())
        result['actions'].append(action);result['branch_probability'].append(float(p[chosen]))
        current=int(future[chosen]);result['states'].append(current)
        if action==0:break
    audit(program,result)
    return result


def audit(program,proof):
    assert proof['found'] and proof['program_sha256']==program.digest()
    current=proof['memory']*program.observations+proof['observed']
    assert proof['states'][0]==current and proof['actions'][-1]==0
    remaining=proof['horizon'];program.extend(remaining)
    assert np.isclose(proof['model_goal_probability'],program._levels[remaining][current])
    for i,action in enumerate(proof['actions']):
        assert action==int(program._policies[remaining][current])
        successor=proof['states'][i+1];nm,ob=divmod(successor,program.observations)
        assert nm==int(program.next_memory[current,action])
        assert program.outcome_probability[current,action,ob]>0
        if action==0:assert i==len(proof['actions'])-1 and successor==program.goal
        current=successor;remaining-=1
    # Verify the executed first action is supported by the entire probability
    # derivation. A tiny successful witness cannot be reported as certainty.
    assert np.isclose(np.dot(proof['root_outcome_probabilities'],proof['root_continuation_probabilities']),
                      proof['model_goal_probability'],rtol=2e-6,atol=1e-12)
    return True
