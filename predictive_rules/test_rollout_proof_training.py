import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.rollout_proof_training import objective
from predictive_nesy.visual_graph_subgoal_training import objective as observed_premise_objective

torch.set_num_threads(2);torch.manual_seed(918)
model=VisualMemorySubgoalModel(Config(states=16,decoder_hidden=64)).train()
b=3;codes=torch.randint(0,64,(b,11,49));actions=torch.randint(0,17,(b,10))
data=dict(task=torch.tensor([0,1,0]),start=torch.zeros(b,dtype=torch.long),history=codes[:,:1].clone(),
    previous_actions=torch.full((b,1),17),history_length=torch.ones(b,dtype=torch.long),targets=codes,actions=actions,
    action_valid=torch.ones(b,10,dtype=torch.bool),state_valid=torch.ones(b,11,dtype=torch.bool),
    supervised=torch.ones(b,10,dtype=torch.bool),goal_index=torch.full((b,),10),episode_id=['a','b','c'])
changed={**data,'targets':codes.clone()};changed['targets'][:,1:]=torch.randint(0,64,(b,10,49))
loss,first=objective(model,data);_,second=objective(model,changed)
assert first['proof_action_imitation']==second['proof_action_imitation']
assert first['proof_action_accuracy']==second['proof_action_accuracy']
# The actual dynamics supervision must still notice a changed observed future.
assert first['one_step_dynamics']!=second['one_step_dynamics']
_,old_first=observed_premise_objective(model,data);_,old_second=observed_premise_objective(model,changed)
assert old_first['proof_action_imitation']!=old_second['proof_action_imitation']
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
print(dict(passed=True,new_proof_loss_future_image_invariant=True,old_proof_loss_changed=True,
    actual_future_still_supervises_dynamics=True,finite_gradients=True))
