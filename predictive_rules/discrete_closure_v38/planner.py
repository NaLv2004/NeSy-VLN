"""Pure integer rule derivation. No torch, images, map, learned action score."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass
class Program:
    next_state: np.ndarray
    confidence: np.ndarray
    goal: int
    observations: int = 64

    def validate(self):
        t=self.next_state
        assert np.issubdtype(t.dtype,np.integer) and t.ndim==2 and t.shape[1]==17
        assert t.min()>=0 and t.max()<len(t) and 0<=self.goal<len(t)
        assert t.shape==self.confidence.shape and np.isfinite(self.confidence).all()
        return self

    def digest(self):
        return hashlib.sha256(self.next_state.tobytes()+self.confidence.tobytes()+str(self.goal).encode()).hexdigest()


def plan(program, memory, observed, horizon=32):
    """All branches must end in STOP at the generated goal. STOP never crosses
    a simulation termination boundary. Observation and memory are independent
    premises: fresh observation DOES NOT replace or re-encode internal memory.
    """
    program.validate()
    current=int(memory)*program.observations+int(observed)
    t=program.next_state
    cost=np.log(np.maximum(program.confidence,1e-12))-0.01
    stop=np.where(t[:,0]==program.goal,cost[:,0],-np.inf)
    levels=[stop]
    policies=[np.zeros(len(t),dtype=np.int64)]
    for _ in range(1,horizon):
        candidate=cost[:,1:]+levels[-1][t[:,1:]]
        action=candidate.argmax(-1)+1
        scores=candidate[np.arange(len(t)),action-1]
        choose_stop=stop>=scores
        levels.append(np.where(choose_stop,stop,scores))
        policies.append(np.where(choose_stop,0,action))
    result=dict(found=bool(np.isfinite(levels[-1][current])),goal=program.goal,
                memory=int(memory),observed=int(observed),states=[current],actions=[],program_sha256=program.digest())
    if not result['found']:return result
    for depth in range(horizon-1,-1,-1):
        action=int(policies[depth][current]);current=int(t[current,action])
        result['actions'].append(action);result['states'].append(current)
        if action==0:break
    audit(program,result)
    return result


def audit(program,proof):
    assert proof['found'] and proof['states'][0]==proof['memory']*program.observations+proof['observed']
    state=proof['states'][0]
    for i,action in enumerate(proof['actions']):
        state=int(program.next_state[state,action]);assert state==proof['states'][i+1]
        if action==0:assert i==len(proof['actions'])-1
    assert proof['actions'][-1]==0 and state==program.goal
    return True
