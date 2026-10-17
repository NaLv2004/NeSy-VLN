"""Small CPU structural/gradient checks, not repeated navigation rollouts."""
import numpy as np
import torch
from model import FactoredWorld,hard
from planner import Program,plan,audit

torch.set_num_threads(2);torch.manual_seed(36)
m=FactoredWorld()
b=2;t=12
data=dict(features=torch.randn(b,t,49,128)*.1,previous=torch.randint(0,18,(b,t)),
    actions=torch.randint(0,17,(b,t-1)),valid=torch.ones(b,t-1,dtype=torch.bool),
    start=torch.tensor([0,2]),end=torch.tensor([11,11]),task=torch.tensor([0,1]),
    branch_features=torch.randn(b,8,49,128)*.1,branch_previous=torch.randint(0,18,(b,8)),
    successors=torch.randn(b,17,49,128)*.1,branch_t=torch.tensor([5,7]),branch_start=torch.tensor([2,3]),branch_task=torch.tensor([0,1]))
loss,metrics=m(data);assert torch.isfinite(loss)
loss.backward()
missing=[name for name,p in m.named_parameters() if p.grad is None]
assert not missing,missing
assert all(torch.isfinite(p.grad).all() for p in m.parameters())
for prefix in ['visual.','observation_definitions','memory.','memory_definitions','rule.','goal.']:
    assert sum(float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(prefix))>0,prefix
with torch.no_grad():
    x=data['features'];o=hard(m.observe(x))
    before=m.context_sequence(x,o,data['previous'],data['task'])[:,:4].clone()
    altered=x.clone();altered[:,4:]=torch.randn_like(altered[:,4:])*10
    after=m.context_sequence(altered,hard(m.observe(altered)),data['previous'],data['task'])[:,:4]
    assert torch.equal(before,after),'Future observation leak into causal rule context'
    output=m.compile(before[:1,-1]);assert output['next_state'].shape==(2048,17)
    Program(**output).validate()
# A proof must use both premises, and table intervention must alter its action.
table=np.tile(np.arange(8)[:,None],(1,17));confidence=np.full((8,17),.9,np.float32)
table[0,1]=3;table[3,0]=7;table[4,2]=3
p=Program(table,confidence,7,observations=4)
assert plan(p,0,0,4)['actions']==[1,0]
assert plan(p,1,0,4)['actions']==[2,0]
assert not plan(p,0,1,4)['found']
table[0,1]=0;assert not plan(p,0,0,4)['found']
print({'status':'passed','all_parameters_have_finite_gradients':True,'causal_context_checked':True,
    'memory_not_overwritten_by_observation':True,'rule_intervention_changes_action':True,
    'full_table_shape':[2048,17],'loss':float(loss.detach())})
