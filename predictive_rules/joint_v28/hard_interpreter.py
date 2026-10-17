"""NumPy-only observation-conditioned transitions and reverse goal planning."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class Program:
    next_state: np.ndarray
    confidence: np.ndarray
    correction: np.ndarray
    goal: int
    start: int

    def validate(self):
        if self.next_state.shape != (512, 17) or self.confidence.shape != (512, 17) or self.correction.shape != (512, 512):
            raise ValueError('Invalid generated rule shapes')
        for data in (self.next_state, self.correction):
            if data.dtype.kind not in 'iu' or ((data<0)|(data>=512)).any():
                raise ValueError('Only hard categorical transitions are executable')
        if not 0 <= self.goal < 512 or not 0 <= self.start < 32:
            raise ValueError('Invalid goal or generation time')
        if not np.isfinite(self.confidence).all() or (self.confidence<=0).any() or (self.confidence>1).any():
            raise ValueError('Invalid rule confidence')
        for data in (self.next_state, self.confidence, self.correction):data.flags.writeable=False
        return self

    def digest(self):
        h=hashlib.sha256()
        for data in (self.next_state,self.confidence,self.correction):h.update(data.tobytes())
        h.update(str((self.goal,self.start)).encode());return h.hexdigest()


def export(generated, index=0):
    table=generated['transitions'][index].detach().float().cpu().numpy()
    return Program(table.argmax(-1).astype(np.int16),table.max(-1),
                   generated['correction_table'][index].detach().cpu().numpy().astype(np.int16),
                   int(generated['subgoal_logits'][index].argmax(-1)),int(generated['start'])).validate()


def corrected_state(program, predicted, observed):
    if type(predicted) is not int or type(observed) is not int or not 0<=predicted<512 or not 0<=observed<512:
        raise TypeError('The interpreter accepts a single scalar observation ID, never image tokens/features')
    return int(program.correction[predicted,observed])


def plan(program, predicted, observed, horizon=10):
    program.validate()
    if not 1 <= horizon <= 10:raise ValueError('Invalid proof horizon')
    current=corrected_state(program,predicted,observed)
    successor=program.next_state
    scale=horizon+1
    confidence=np.rint(np.log(program.confidence.astype(np.float64))*10000).astype(np.int64)*scale-1
    impossible=-10**15;terminal=np.full(512,impossible,np.int64);terminal[program.goal]=0
    support=terminal;policies=[None]
    for depth in range(1,horizon+1):
        scores=confidence+support[successor]
        scores[:,0]=confidence[:,0]+terminal[successor[:,0]]
        selected=scores.argmax(-1);policies.append(selected);support=scores[np.arange(512),selected]
    base=dict(goal=program.goal,observed=observed,predicted=predicted,corrected=current,
              horizon=horizon,program_sha256=program.digest(),hard_observation_conditioning=True)
    if support[current]<impossible//2:return dict(**base,found=False,actions=[],states=[current])
    states=[current];actions=[]
    for remaining in range(horizon,0,-1):
        action=int(policies[remaining][current]);current=int(successor[current,action])
        actions.append(action);states.append(current)
        if action==0:break
    assert current==program.goal
    return dict(**base,found=True,actions=actions,states=states)
