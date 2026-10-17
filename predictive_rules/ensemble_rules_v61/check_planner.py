"""Algebraic checks, not a navigation-performance evaluation."""
import json,sys
from pathlib import Path
import numpy as np
from robust_planner import Program,plan
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'goal_equivalence_v57'))
from planner import Program as SingleProgram,plan as single_plan
rng=np.random.default_rng(611017)
maximum_error=0.
for case in range(12):
    t=rng.integers(0,3,size=(12,17));p=rng.dirichlet(np.ones(4),size=(12,17));goal=int(rng.integers(12))
    a=Program(t,p[None],goal,4).validate();b=SingleProgram(t,p,goal,4).validate()
    for h in (1,2,5,12):
        a.extend(h);b.extend(h)
        maximum_error=max(maximum_error,float(np.max(abs(a._levels[h]-b._levels[h]))))
        assert np.allclose(a._levels[h],b._levels[h],atol=1e-14)
        assert np.array_equal(a._policies[h],b._policies[h])
        for state in (0,5,11):
            x=plan(a,state//4,state%4,h);y=single_plan(b,state//4,state%4,h)
            assert x['actions']==y['actions'] and x['states']==y['states']
# Model0 optimistically favors action1; model1 disagrees. Action2 is .6 in both.
t=np.zeros((4,17),dtype=np.int64);p=np.zeros((2,4,17,2));p[...,0]=1
for state in range(4):t[state,0]=state//2;p[:,state,0]=np.eye(2)[state%2]
t[0,1:3]=1;p[0,0,1]=[.05,.95];p[1,0,1]=[.65,.35];p[:,0,2]=[.4,.6]
robust=Program(t,p,3,2).validate();proof=plan(robust,0,0,2)
optimistic=single_plan(SingleProgram(t,p[0],3,2),0,0,2)
mean=single_plan(SingleProgram(t,p.mean(0),3,2),0,0,2)
assert proof['actions'][0]==2 and optimistic['actions'][0]==mean['actions'][0]==1
assert np.isclose(proof['model_goal_probability'],.6)
def brute(state,h):
    if h==0:return 0.
    choices=[]
    for action in range(17):
        vals=[]
        for member in range(2):
            vals.append(sum(p[member,state,action,o]*((int(t[state,action])*2+o==3) if action==0 else brute(int(t[state,action])*2+o,h-1)) for o in range(2)))
        choices.append(min(vals))
    return max(choices)
assert np.isclose(brute(0,2),proof['model_goal_probability'])
result=dict(one_member_matches_existing_planner=True,maximum_value_error=maximum_error,
    robust_action=proof['actions'][0],mean_action=mean['actions'][0],brute_force_matches=True,
    pure_numpy=True,navigation_tested=False)
out=Path(__file__).with_name('planner_contract.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
