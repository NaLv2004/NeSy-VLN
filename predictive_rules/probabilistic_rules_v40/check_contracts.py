import json
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,hard
from planner import Program,plan,audit

torch.set_num_threads(3);torch.manual_seed(40)
# A modal-only model would retain self loops and falsely say no route exists.
t=np.zeros((3,17),dtype=np.int64)
p=np.eye(3,dtype=np.float32)[:,None,:].repeat(17,1)
p[0,1]=[.51,0,.49];p[0,2]=[.8,0,.2]
program=Program(t,p,goal=2,observations=3)
proof=plan(program,0,0,horizon=2)
assert proof['actions']==[1,0] and abs(proof['model_goal_probability']-.49)<1e-6
assert int(p[0,1].argmax())==0
proof3=plan(program,0,0,horizon=3)
assert abs(proof3['model_goal_probability']-(1-.51**2))<1e-6
assert not plan(program,0,0,horizon=1)['found']
bad=dict(proof,model_goal_probability=.99)
try:audit(program,bad)
except AssertionError:pass
else:raise AssertionError('An exaggerated proof probability was accepted')

model=FactoredWorld().eval();c=model.cfg
o=F.one_hot(torch.randint(c.observations,(2,6)),c.observations).float()
a=torch.randint(c.actions,(2,5));previous=torch.cat([torch.full((2,1),17),a],1)
ml,_,_=model.memory_sequence(o,previous)
for i in range(5):
    expected,_=model.memory_step(hard(ml[:,i]),o[:,i],a[:,i])
    assert torch.equal(expected,ml[:,i+1])
changed=o.clone();changed[:,3]=changed[:,3].roll(1,-1)
changed_m,_,_=model.memory_sequence(changed,previous)
assert torch.equal(ml[:,:4],changed_m[:,:4]),'Future observation corrected prior memory'
context=torch.randn(1,c.context);m=F.one_hot(torch.tensor([[3]]),c.memories).float()
observed=F.one_hot(torch.tensor([[8]]),c.observations).float();action=torch.tensor([[4]])
pred_m,_=model.transition(context,m,observed,action)
direct_m,_=model.memory_step(m,observed,action)
assert torch.equal(pred_m,direct_m)

with torch.no_grad():
    # Arbitrary opaque ID, not an assignment of a named navigation effect.
    model.operator_class.weight[5]=torch.tensor([-20.,20.])
    compiled=model.compile(context,torch.tensor([0]))
compiled_program=Program(**compiled).validate()
ids=np.arange(c.memories*c.observations)%c.observations
assert np.array_equal(compiled['outcome_probability'][:,5,:],np.eye(c.observations,dtype=np.float32)[ids])
assert not any(k in compiled for k in ('value','reward','action_logits','policy'))
print(json.dumps(dict(causal_hard_memory=True,future_observation_cannot_correct_memory=True,
    probabilistic_proof_exact=True,modal_pruning_failure_demonstrated=True,
    forged_success_probability_rejected=True,opaque_empirical_invariant_preserved=True,
    no_neural_actor_or_value_output=True,complete_rule_rows=int(compiled['next_memory'].size))))
