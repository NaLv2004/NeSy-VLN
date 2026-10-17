import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config,export_program
from predictive_nesy.visual_graph_subgoal_training import objective
torch.set_num_threads(2);torch.manual_seed(951)
m=VisualMemorySubgoalModel(Config(states=16,decoder_hidden=64));b=4
codes=torch.randint(0,64,(b,11,49));d=dict(task=torch.tensor([0,1,0,1]),start=torch.zeros(b,dtype=torch.long),
    history=codes[:,:1],previous_actions=torch.full((b,1),17),history_length=torch.ones(b,dtype=torch.long),
    targets=codes,actions=torch.randint(0,17,(b,10)),action_valid=torch.ones(b,10,dtype=torch.bool),
    state_valid=torch.ones(b,11,dtype=torch.bool),supervised=torch.ones(b,10,dtype=torch.bool),
    goal_index=torch.full((b,),10),episode_id=list('abcd'))
def program():return m.generate(d['task'],d['history'],d['previous_actions'],d['history_length'],start=0)
before=export_program(program()).digest()
la,a=objective(m,d,proof_from_predicted_state=True)
lb,c=objective(m,d,proof_from_predicted_state=True,proof_only_if_goal_correct=True)
assert export_program(program()).digest()==before
assert c['proof_action_accuracy']==a['proof_action_accuracy']
assert c['proof_action_imitation']<=a['proof_action_imitation']+1e-6
for key in a:
    if key not in ['loss','proof_action_imitation']:assert abs(a[key]-c[key])<1e-6,key
assert abs(float(lb.detach()-la.detach())-(c['proof_action_imitation']-a['proof_action_imitation']))<1e-4
lb.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print(dict(passed=True,generated_program_identical=True,derived_actions_identical=True,
    only_loss_mask_changed=True,future_goal_never_substituted=True))
