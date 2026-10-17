"""Shortest maximum-likelihood rule proof to the exact generated goal.

All action outcomes remain available. The product is the probability of one
specified latent outcome path, NOT a policy's overall physical success rate.
Exact ties prefer fewer actions. No neural module, map or state reward exists.
"""
from dataclasses import dataclass,field
import hashlib
import numpy as np

@dataclass
class Program:
    next_memory:np.ndarray
    outcome_probability:np.ndarray
    goal:int
    observations:int=64
    _levels:list=field(default_factory=list,repr=False)
    _policies:list=field(default_factory=list,repr=False)
    _lengths:list=field(default_factory=list,repr=False)
    _branches:list=field(default_factory=list,repr=False)
    _digest:str=field(default='',repr=False)
    _probability:np.ndarray=field(default=None,repr=False)
    def validate(self):
        t=self.next_memory;p=self.outcome_probability;n=len(t)
        assert np.issubdtype(t.dtype,np.integer) and t.ndim==2 and t.shape[1]==17
        assert t.min()>=0 and t.max()<n//self.observations
        assert p.shape==(*t.shape,self.observations) and np.isfinite(p).all()
        assert p.min()>=0 and np.allclose(p.sum(-1),1,atol=2e-6)
        assert 0<=self.goal<n
        if self._probability is None:
            p64=p.astype(np.float64)
            self._probability=p64/p64.sum(-1,keepdims=True)
        return self
    def digest(self):
        if not self._digest:
            self._digest=hashlib.sha256(self.next_memory.tobytes()+self.outcome_probability.tobytes()+str(self.goal).encode()).hexdigest()
        return self._digest
    def extend(self,horizon):
        if not self._levels:
            self.validate();n=len(self.next_memory)
            self._levels=[np.zeros(n,np.float64)]
            self._lengths=[np.full(n,1000,np.int16)]
            self._policies=[np.zeros(n,np.int16)]
            self._branches=[np.zeros((n,17),np.int16)]
        gm,go=divmod(self.goal,self.observations)
        stop=np.where(self.next_memory[:,0]==gm,self._probability[:,0,go],0.)
        while len(self._levels)<=horizon:
            previous=self._levels[-1].reshape(-1,self.observations)
            lengths=self._lengths[-1].reshape(-1,self.observations)
            mass=self._probability*previous[self.next_memory]
            q=mass.max(-1)
            candidate_lengths=1+lengths[self.next_memory]
            # Only exact equal-probability outcomes are tie-broken by length.
            branch=np.where(mass==q[...,None],candidate_lengths,2000).argmin(-1)
            action_lengths=np.take_along_axis(candidate_lengths,branch[...,None],-1)[...,0]
            q[:,0]=stop;action_lengths[:,0]=1;branch[:,0]=go
            best=q.max(-1)
            action=np.where(q==best[:,None],action_lengths,2000).argmin(-1)
            rows=np.arange(len(q))
            self._levels.append(best)
            self._lengths.append(action_lengths[rows,action].astype(np.int16))
            self._policies.append(action.astype(np.int16));self._branches.append(branch.astype(np.int16))

def plan(program,memory,observed,horizon=32):
    assert 1<=horizon<=32
    program.extend(horizon);initial=int(memory)*program.observations+int(observed)
    value=float(program._levels[horizon][initial])
    proof=dict(found=value>0,goal=program.goal,memory=int(memory),observed=int(observed),horizon=horizon,
        maximum_path_probability=value,states=[initial],actions=[],branch_probability=[],
        program_sha256=program.digest(),planner='shortest_maximum_likelihood_path',
        derivation='Max over all action/outcome paths of the probability product; exact ties prefer shorter proofs. Score is not a physical success guarantee.')
    if not proof['found']:return proof
    current=initial
    for remaining in range(horizon,0,-1):
        action=int(program._policies[remaining][current]);ob=int(program._branches[remaining][current,action])
        nm=int(program.next_memory[current,action]);p=program._probability[current,action]
        if remaining==horizon:proof.update(next_memory=nm,predicted_observation=int(p.argmax()),root_outcome_probabilities=p.tolist())
        proof['actions'].append(action);proof['branch_probability'].append(float(p[ob]))
        current=nm*program.observations+ob;proof['states'].append(current)
        if action==0:break
    audit(program,proof)
    return proof

def audit(program,proof):
    assert proof['found'] and proof['program_sha256']==program.digest()
    assert proof['planner']=='shortest_maximum_likelihood_path'
    current=proof['memory']*program.observations+proof['observed'];remaining=proof['horizon']
    program.extend(remaining);assert proof['states'][0]==current and proof['actions'][-1]==0
    assert len(proof['actions'])==int(program._lengths[remaining][current])
    assert np.isclose(proof['maximum_path_probability'],program._levels[remaining][current],rtol=1e-9,atol=1e-15)
    probabilities=[]
    for i,action in enumerate(proof['actions']):
        assert action==int(program._policies[remaining][current])
        nm,ob=divmod(proof['states'][i+1],program.observations)
        assert nm==int(program.next_memory[current,action]) and ob==int(program._branches[remaining][current,action])
        p=float(program._probability[current,action,ob]);assert p>0
        probabilities.append(p)
        if action==0:assert i==len(proof['actions'])-1 and proof['states'][i+1]==program.goal
        current=proof['states'][i+1];remaining-=1
    assert np.allclose(probabilities,proof['branch_probability'],rtol=0,atol=0)
    assert np.isclose(np.prod(probabilities),proof['maximum_path_probability'],rtol=1e-8,atol=1e-15)
    return True
