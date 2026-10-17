import numpy as np
from planner import Program,plan
from root_planner import derive
rng=np.random.default_rng(69017);errors=[]
for _ in range(24):
    nm=rng.integers(0,2,size=(128,17));p=rng.uniform(.01,1,size=(128,17,64));p/=p.sum(-1,keepdims=True)
    # There must be at least one exact terminal preimage.
    nm[:,0]=0
    program=Program(nm,p,goal=3).validate();m=int(rng.integers(2));o=int(rng.integers(64));h=int(rng.integers(2,12))
    original=plan(program,m,o,h);new=derive(program,m,o,nm[m*64+o],p[m*64+o],h)
    assert original['actions'][0]==new['actions'][0]
    errors.append(abs(original['model_goal_probability']-new['model_goal_probability']))
print(dict(cases=len(errors),max_probability_error=max(errors),all_first_actions_identical=True,optimizer_updates=0))
