"""A task goal gates symbolic termination without replacing the subgoal."""
import numpy as np
from predictive_nesy.graph_subgoal_interpreter import Program,plan
rng=np.random.default_rng(812)
for _ in range(64):
    table=rng.dirichlet(np.ones(8),size=(8,5)).astype(np.float32)
    goal=np.array([rng.integers(8)]);p=Program(table,np.eye(8),goal,0)
    start=np.array([rng.integers(8)])
    old=plan(p,start,4);same=plan(p,start,4,terminal_goal=goal)
    assert old==same
    other=np.array([(int(goal[0])+1)%8]);other_plan=plan(p,start,4,terminal_goal=other)
    assert not other_plan['found'] or (len(other_plan['actions'])==4 and 0 not in other_plan['actions'])
    assert np.array_equal(p.subgoal,goal)
# One movement reaches an intermediate state. STOP stays there. A separate
# task goal must not turn that into a successful terminal proof.
t=np.zeros((3,2,3),np.float32)
t[0,0,0]=1;t[0,1,1]=1;t[1,0,1]=1;t[1,1,1]=1;t[2,:,2]=1
p=Program(t,np.eye(3),np.array([1]),0)
assert plan(p,np.array([0]),3)['actions']==[1,0]
r=plan(p,np.array([0]),3,terminal_goal=np.array([2]))
assert r['actions']==[1,1,1] and r['states'][-1]==[1]
print(dict(passed=True,legacy_cases=64,distinct_task_goal_blocks_intermediate_STOP=True,unmodified_subgoal=True))
