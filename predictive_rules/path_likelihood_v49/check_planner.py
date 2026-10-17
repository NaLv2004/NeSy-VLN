from functools import lru_cache
import json
import numpy as np
from planner import Program,plan

t=np.zeros((4,17),np.int64);p=np.zeros((4,17,4),np.float32);p[:,:,3]=1
for s in range(4):p[s,0]=np.eye(4)[s]
p[0,1]=[1,0,0,0];p[0,2]=[0,0,.6,.4];p[0,3]=[0,.9,0,.1];p[1,2]=[0,0,.9,.1]
program=Program(t,p,goal=2,observations=4)
@lru_cache(None)
def reference(state,horizon):
    if horizon==0:return (0.,1000)
    choices=[]
    for action in range(17):
        if action==0:choices.append((float(p[state,0,2]),1));continue
        for outcome in range(4):
            probability,length=reference(outcome,horizon-1)
            choices.append((float(p[state,action,outcome])*probability,1+length))
    return max(choices,key=lambda x:(x[0],-x[1]))
for horizon in range(1,7):
    program.extend(horizon)
    for state in range(4):
        expected=reference(state,horizon)
        assert np.isclose(program._levels[horizon][state],expected[0])
        if expected[0]>0:assert program._lengths[horizon][state]==expected[1]
proof=plan(program,0,0,32)
assert proof['actions']==[3,2,0]
assert np.isclose(proof['maximum_path_probability'],.81)
assert not plan(program,0,3,32)['found']
print(json.dumps(dict(passed=True,scalar_reference_cases=24,zero_cost_self_loop_does_not_delay=True,
    path=proof['actions'],path_probability=proof['maximum_path_probability'])))
