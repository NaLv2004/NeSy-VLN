import torch
from predictive_nesy.graph_subgoal_training import transition
from predictive_nesy.graph_subgoal_model import straight_through

device='cuda';torch.set_num_threads(2)
table=torch.tensor([[[[.4999,.5001]],[[.6,.4]]]],device=device,requires_grad=True)
state=torch.tensor([[1.,0.]],device=device,requires_grad=True);action=torch.zeros(1,dtype=torch.long,device=device)
with torch.autocast('cuda',dtype=torch.bfloat16):
    old=torch.einsum('bs,bsk->bk',state,table[:,:,0])
    new=transition({'transitions':table},state,action)
assert old.argmax(-1).item()==0 and new.argmax(-1).item()==1
new[0,1].backward()
assert state.grad is not None and table.grad is not None and torch.isfinite(table.grad).all()
torch.manual_seed(533)
probabilities=torch.randn(128,128,17,128,device=device).softmax(-1)
ids=torch.randint(0,128,(128,),device=device);states=torch.nn.functional.one_hot(ids,128).float()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for _ in range(10):
        actions=torch.randint(0,17,(128,),device=device)
        exact=probabilities[torch.arange(128,device=device),ids,actions]
        predicted=transition({'transitions':probabilities},states,actions)
        assert torch.equal(predicted,exact)
        ids=exact.argmax(-1);states=straight_through(predicted)
print(dict(passed=True,deliberate_bf16_tie_corrected=True,exact_transitions_compared=1280,gradients_preserved=True))
