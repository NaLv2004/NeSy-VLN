import pytest
import torch
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.spatial_token_automaton import SpatialTokenAutomaton


def test_shared_vocabulary_and_integer_only_input():
    m=SpatialTokenAutomaton(patches=9,vocabulary=8,width=8)
    x=torch.tensor([[2,3,2,4,2,1,0,2,7]])
    embedded=m.embed(x).reshape(1,9,8)
    assert torch.equal(embedded[:,0],embedded[:,2])
    with pytest.raises(TypeError):m.embed(x.float())
    with pytest.raises(ValueError):m.embed(torch.full_like(x,8))


def test_spatial_reader_preserves_rule_proofs_and_has_no_action_bypass():
    torch.manual_seed(5900)
    m=SpatialTokenAutomaton(patches=9,vocabulary=8,width=8,task_banks=True,context_bridge=True)
    x=torch.randint(8,(2,9));goals=torch.tensor([0,1]);program=m.generate(goals,x[:,None],0)
    _,prior=m.initial_state(2);actions=torch.full((2,),17,dtype=torch.long)
    r=m.step(program,x,prior,actions,0)
    proof=m.proof(program,r['observed_ids'],prior,actions,r['state_values'],0)
    assert torch.equal(proof['probabilities'],r['probabilities'])
    (-r['probabilities'][:,5].log().mean()).backward()
    assert m.observation.conv[0].weight.grad.norm()>0
    assert m.conclusions.grad.norm()>0
    with torch.no_grad():m.conclusions.zero_();m.conclusion_basis.zero_()
    program=m.generate(goals,x[:,None],0)
    p=m.step(program,x,prior,actions,0)['probabilities']
    assert torch.allclose(p,torch.ones_like(p)/17,atol=1e-7)


def test_matched_initialization_copies_rules_and_generator_only():
    kwargs=dict(patches=9,vocabulary=8,width=8,task_banks=True,context_bridge=True)
    base=ProbabilisticTokenAutomaton(**kwargs);spatial=SpatialTokenAutomaton(**kwargs)
    copied=spatial.initialize_common_program(base.state_dict())
    assert 'conclusions' in copied and 'history.weight_ih_l0' in copied
    for key in copied:assert torch.equal(base.state_dict()[key],spatial.state_dict()[key])
