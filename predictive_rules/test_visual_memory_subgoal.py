import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config,export_program
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel
from predictive_nesy.visual_memory_subgoal_training import objective,transition_objective
from predictive_nesy.gate import NeuralBoundaryGate


def run(device='cpu'):
    torch.manual_seed(76);torch.set_num_threads(2)
    config=Config(states=16,decoder_hidden=32)
    original=VisualGraphSubgoalModel(config).to(device).eval()
    model=VisualMemorySubgoalModel(config).to(device).eval()
    model.load_state_dict(original.state_dict(),strict=False)
    frames=torch.randint(0,64,(2,11,49),device=device)
    tasks=torch.tensor([0,1],device=device);prior=torch.full((2,1),17,device=device)
    lengths=torch.ones(2,dtype=torch.long,device=device)
    p=model.generate(tasks,frames[:,:1],prior,lengths,start=0)
    old=original.generate(tasks,frames[:,:1],prior,lengths,start=0)
    assert torch.equal(model.ground(p,frames[:,0]),original.ground(old,frames[:,0]))
    assert torch.equal(p['transitions'],old['transitions'])
    # Use nonzero memory influence to test stutter invariance meaningfully.
    torch.nn.init.normal_(model.memory_projection.weight,std=.04)
    p=model.generate(tasks,frames[:,:1],prior,lengths,start=0)
    repeated=frames.clone();repeated[:,5:]=repeated[:,4:5]
    actions=torch.randint(0,17,(2,10),device=device)
    states,memory,_,_=model.encode_sequence(p,repeated)
    assert torch.equal(states[:,4:5].expand(-1,7,-1),states[:,4:])
    labels=model.training_targets(p,repeated,actions)
    assert torch.equal(labels,model.training_targets(p,repeated,torch.zeros_like(actions)))
    permuted=frames.clone();permuted[:,1:4]=permuted[:,1:4].flip(1)
    _,other_memory,_,_=model.encode_sequence(p,permuted)
    assert not torch.equal(memory,other_memory)
    exported=export_program(p)
    assert set(exported.__dict__)=={'definitions','transitions','subgoal','start','interval'}
    data=dict(task=tasks,start=torch.zeros(2,dtype=torch.long,device=device),history=frames,
        previous_actions=torch.full((2,11),17,device=device),history_length=lengths,targets=repeated,actions=actions,
        action_valid=torch.ones(2,10,dtype=torch.bool,device=device),state_valid=torch.ones(2,11,dtype=torch.bool,device=device),
        supervised=torch.ones(2,10,dtype=torch.bool,device=device),goal_index=torch.full((2,),10,device=device),episode_id=['a','b'])
    model.train()
    loss,_=objective(model,data);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    bank=dict(task=tasks,start=0,offset=2,history=frames[:,:1],previous_actions=prior,
        prefix_codes=frames[:,:3],prefix_actions=actions[:,:2],successors=torch.randint(0,64,(2,17,49),device=device),episode_ids=['a','b'])
    model.zero_grad(set_to_none=True);loss,_=transition_objective(model,bank);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    gate=NeuralBoundaryGate([model]);caught=False
    try:model.generate(tasks,frames[:,:1],prior,lengths,start=0)
    except RuntimeError:caught=True
    finally:gate.close()
    assert caught
    print(dict(passed=8,device=device,finite_gradients=True,stutter_invariant=True,future_actions_ignored=True,
        no_grounding_memory_exported=True,initial_function_preserved=True))


if __name__=='__main__':
    import sys
    run('cuda' if '--cuda' in sys.argv else 'cpu')
