import json
import numpy as np
from planner import Program,plan,audit

memory=np.zeros((4,17),np.int64);p=np.zeros((4,17,4),np.float32)
for state in range(4):p[state,:,state]=1
p[0,1]=[.95,0,.05,0];p[0,2]=[0,0,.6,.4]
results={}
for discount,expected in [(1.,1),(.95,2)]:
    program=Program(memory,p,2,observations=4,discount=discount).validate()
    proof=plan(program,0,0,32);audit(program,proof)
    assert proof['actions'][0]==expected
    assert proof['actions'][-1]==0 and proof['states'][-1]==2
    # Independent scalar Bellman recursion verifies the root value.
    value=np.zeros(4)
    for _ in range(32):
        q=discount*np.einsum('sao,o->sa',p,value);q[:,0]=p[:,0,2]
        value=q.max(-1)
    assert np.isclose(value[0],proof['discounted_goal_score'])
    results[str(discount)]=dict(action=expected,score=proof['discounted_goal_score'])
print(json.dumps(dict(passed=True,uncertainty_loop_fixture=results,unchanged_generated_goal=True)))
