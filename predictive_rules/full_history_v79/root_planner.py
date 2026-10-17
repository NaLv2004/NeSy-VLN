"""Symbolic one-step history-conditioned rows plus finite-table continuation.

No neural scores/actions/heuristic are used. Root transition probabilities come
from the state predictor; all subsequent continuation probabilities are derived
by Bellman recursion from the existing compiled discrete transition table.
"""
import hashlib
import numpy as np
from planner import plan

def derive(program,memory,observed,next_memory,outcomes,horizon):
    program.extend(max(0,horizon-1))
    p=np.asarray(outcomes,dtype=np.float64);p=p/p.sum(-1,keepdims=True)
    nm=np.asarray(next_memory,dtype=np.int64);assert p.shape==(17,64) and nm.shape==(17,)
    assert np.isfinite(p).all() and p.min()>=0
    future=nm[:,None]*64+np.arange(64)[None]
    continuation=program._levels[horizon-1][future].copy()
    failure=program._failures[horizon-1][future].copy()
    continuation[0]=(future[0]==program.goal).astype(float);failure[0]=1-continuation[0]
    q=(p*continuation).sum(-1);f=(p*failure).sum(-1)
    a=int(q.argmax() if q.max()<=.5 else f.argmin())
    value=float(q[a]);found=value>0
    result=dict(found=found,goal=program.goal,memory=int(memory),observed=int(observed),horizon=horizon,
        model_goal_probability=value,model_failure_probability=float(f[a]),
        program_sha256=program.digest(),root_next_memory=nm.tolist(),root_transition_probabilities=p.tolist(),
        root_rows_sha256=hashlib.sha256(nm.tobytes()+p.tobytes()).hexdigest(),
        derivation='Symbolic Bellman comparison of neural history-conditioned next-state rows followed by compiled-table continuation',
        neural_action_scores_used=False,neural_lookahead_depth=1)
    if not found:return result
    chosen=int((p[a]*continuation[a]).argmax());successor=int(future[a,chosen]);initial=int(memory)*64+int(observed)
    result.update(actions=[a],states=[initial,successor],next_memory=int(nm[a]),predicted_observation=int(p[a].argmax()),
        branch_probability=[float(p[a,chosen])],root_outcome_probabilities=p[a].tolist(),
        root_continuation_probabilities=continuation[a].tolist(),root_continuation_failures=failure[a].tolist())
    if a!=0:
        assert horizon>1
        tail=plan(program,int(nm[a]),chosen,horizon-1);assert tail['found']
        result['actions']+=tail['actions'];result['states']+=tail['states'][1:];result['branch_probability']+=tail['branch_probability']
    assert result['actions'][-1]==0 and result['states'][-1]==program.goal
    audit_root(program,result)
    return result

def audit_root(program,proof):
    assert proof['program_sha256']==program.digest() and proof['neural_action_scores_used'] is False
    p=np.asarray(proof['root_transition_probabilities']);nm=np.asarray(proof['root_next_memory'],dtype=np.int64)
    assert hashlib.sha256(nm.tobytes()+p.tobytes()).hexdigest()==proof['root_rows_sha256']
    h=proof['horizon'];program.extend(h-1);future=nm[:,None]*64+np.arange(64)[None]
    c=program._levels[h-1][future].copy();f=program._failures[h-1][future].copy()
    c[0]=(future[0]==program.goal);f[0]=(future[0]!=program.goal)
    q=(p*c).sum(-1);fail=(p*f).sum(-1)
    a=int(q.argmax() if q.max()<=.5 else fail.argmin())
    assert a==proof['actions'][0] and np.isclose(q[a],proof['model_goal_probability'])
    assert np.isclose(q[a]+fail[a],1)
    return True

