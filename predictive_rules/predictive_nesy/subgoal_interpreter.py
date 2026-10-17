"""Exact anonymous-goal reachability. No neural network or task-return score."""
from dataclasses import dataclass
import hashlib
import numpy as np

FIELDS=('parents','transitions','definitions','subgoal')


@dataclass(frozen=True)
class Program:
    parents:np.ndarray
    transitions:np.ndarray
    definitions:np.ndarray
    subgoal:np.ndarray
    start:int
    interval:int=10

    def validate(self):
        f,a,k,k2,k3=self.transitions.shape
        assert k==k2==k3 and self.parents.shape==(f,2) and self.subgoal.shape==(f,)
        assert self.parents.dtype.kind in 'iu' and ((0<=self.parents)&(self.parents<f)).all()
        assert self.subgoal.dtype.kind in 'iu' and ((0<=self.subgoal)&(self.subgoal<k)).all()
        assert self.definitions.shape[:2]==(f,k)
        assert np.isfinite(self.transitions).all() and (self.transitions>=0).all()
        assert np.allclose(self.transitions.sum(-1),1,atol=1e-5)
        assert self.start>=0 and self.start%10==0
        return self

    def digest(self):
        h=hashlib.sha256()
        for field in FIELDS:h.update(np.ascontiguousarray(getattr(self,field)).tobytes())
        h.update(str((self.start,self.interval)).encode());return h.hexdigest()

    def save(self,path):np.savez_compressed(path,**self.__dict__)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            if set(f.files)!=set(FIELDS)|{'start','interval'}:raise ValueError('Unexpected executable field')
            data={key:f[key] for key in f.files}
        data['start']=int(data['start']);data['interval']=int(data['interval'])
        return cls(**data).validate()


def export_program(generated,index=0):
    if set(generated)!={'parents','transitions','definitions','subgoal_logits','start','context_raw'}:raise ValueError('Unexpected model outputs')
    return Program(parents=generated['parents'][index].detach().argmax(-1).cpu().numpy().astype(np.int64),
        transitions=generated['transitions'][index].detach().float().cpu().numpy().copy(),
        definitions=generated['definitions'][index].detach().float().cpu().numpy().copy(),
        subgoal=generated['subgoal_logits'][index].detach().argmax(-1).cpu().numpy().astype(np.int64),start=generated['start']).validate()


def successors(program,states):
    states=np.asarray(states,dtype=np.int64);one=states.ndim==1
    if one:states=states[None]
    f,a,k,_,_=program.transitions.shape
    left=states[:,program.parents[:,0]];right=states[:,program.parents[:,1]]
    probabilities=program.transitions[np.arange(f)[None,None],np.arange(a)[None,:,None],left[:,None],right[:,None]]
    after=probabilities.argmax(-1);confidence=np.log(probabilities.max(-1).clip(1e-8)).sum(-1)
    return (after[0],confidence[0]) if one else (after,confidence)


def plan(program,initial,horizon=10,beam_width=128):
    program.validate();initial=np.asarray(initial,dtype=np.int64)
    if initial.shape!=program.subgoal.shape or not 1<=horizon<=10:raise ValueError('Invalid planning request')
    states=initial[None];paths=[[]];proofs=[[initial.tolist()]];confidence=np.zeros(1);expanded=0;closest=None
    for depth in range(horizon):
        after,logp=successors(program,states);a=after.shape[1]
        parent=np.repeat(np.arange(len(states)),a);actions=np.tile(np.arange(a),len(states))
        flat=after.reshape(-1,len(initial));distance=(flat!=program.subgoal).sum(-1)
        certainty=(confidence[:,None]+logp).reshape(-1);expanded+=len(flat)
        accepted=(distance==0)&((actions==0)|(depth+1==horizon))
        if accepted.any():
            candidates=np.flatnonzero(accepted);choice=candidates[np.argmax(certainty[candidates])];p=int(parent[choice]);act=int(actions[choice])
            return dict(found=True,actions=paths[p]+[act],states=proofs[p]+[flat[choice].tolist()],
                subgoal=program.subgoal.tolist(),expanded=expanded,planning_horizon=horizon,beam_width=beam_width,
                program_sha256=program.digest(),goal_distance=0)
        order=np.lexsort((-certainty,distance));seen=set();kept=[]
        for index in order:
            if actions[index]==0:continue
            key=tuple(flat[index])
            if key in seen:continue
            seen.add(key);kept.append(int(index))
            if len(kept)==beam_width:break
        if not kept:break
        states=flat[kept];confidence=certainty[kept]
        paths=[paths[int(parent[i])]+[int(actions[i])] for i in kept]
        proofs=[proofs[int(parent[i])]+[flat[i].tolist()] for i in kept]
        closest=dict(actions=paths[0],states=proofs[0],distance=int(distance[kept[0]]))
    return dict(found=False,actions=[],states=[initial.tolist()],subgoal=program.subgoal.tolist(),
        expanded=expanded,planning_horizon=horizon,beam_width=beam_width,program_sha256=program.digest(),
        goal_distance=None if closest is None else closest['distance'],closest_candidate_for_training_only=closest)


class Controller:
    def __init__(self):self.step=0;self.queue=[];self.finished=False;self.program=None;self.current=None

    def revise(self,program,state,*,horizon=10,beam_width=128):
        if self.finished or self.step%10 or self.queue or program.start!=self.step:raise ValueError('Revision outside boundary')
        self.program=program.validate();self.current=np.asarray(state,dtype=np.int64).copy()
        result=plan(program,self.current,horizon,beam_width);self.queue=result['actions'].copy()
        return result

    def next_action(self):
        if self.finished or not self.queue:raise ValueError('No proven symbolic plan')
        action=self.queue.pop(0);before=self.current.copy();after,_=successors(self.program,self.current)
        self.current=after[action]
        if action==0:self.finished=True;self.queue.clear()
        proof=dict(step=self.step,program_start=self.program.start,action=action,before=before.tolist(),after=self.current.tolist(),
            program_sha256=self.program.digest(),subgoal=self.program.subgoal.tolist(),subgoal_reached=bool(np.array_equal(self.current,self.program.subgoal)))
        self.step+=1;return action,proof
