import pytest
import torch
from nesy_vla.token_rule_revision import TokenFacts,RuleRevision,TokenProgramReviser,replay_tree_program,PackedRuleProgram


def trees():
    return [dict(slots=[0,-1,-1],values=[1,-1,-1],false_branch=[1,-1,-1],true_branch=[2,-1,-1],
                 conclusion=[None,[1.,0.],[0.,1.]]),
            dict(slots=[1,-1,-1],values=[2,-1,-1],false_branch=[1,-1,-1],true_branch=[2,-1,-1],
                 conclusion=[None,[0.,1.],[1.,0.]])]


def test_equal_program_reproduces_induced_rules_and_only_true_paths_fire():
    model=TokenProgramReviser((3,4),1,2)
    state=TokenFacts(torch.tensor([1,2]),(3,4))
    revision=model(0,TokenFacts(state.codes[None],(3,4)),0)
    p,proof,_=replay_tree_program(trees(),state,revision,0)
    assert torch.equal(revision.weights,torch.tensor([.5,.5]))
    assert torch.equal(p,torch.tensor([.5,.5]))
    assert proof[0]['leaf']==2 and proof[1]['leaf']==2
    p,proof,_=replay_tree_program(trees(),TokenFacts(torch.tensor([0,2]),(3,4)),RuleRevision(torch.tensor([1.,0.]),0,10),0)
    assert torch.equal(p,torch.tensor([1.,0.])) and proof[0]['leaf']==1


def test_revision_gradient_flows_only_through_weights_not_visual_bypass():
    model=TokenProgramReviser((3,4),1,2)
    state=TokenFacts(torch.tensor([1,2]),(3,4))
    revision=model(0,TokenFacts(state.codes[None],(3,4)),0)
    p,_,_=replay_tree_program(trees(),state,revision,0)
    (-p[0].log()).backward()
    assert model.program.weight.grad.norm()>0
    assert state.codes.grad is None
    with pytest.raises(TypeError):model(0,torch.zeros(1,3,224,224),0)
    with pytest.raises(TypeError):replay_tree_program(trees(),TokenFacts(torch.tensor([1.,2.]),(3,4)),revision,0)


def test_program_is_immutable_for_ten_actions_and_rejects_future_context():
    model=TokenProgramReviser((3,4),1,2);facts=TokenFacts(torch.tensor([[1,2]]),(3,4))
    p=model(0,facts,0)
    replay_tree_program(trees(),TokenFacts(facts.codes[0],(3,4)),p,9)
    with pytest.raises(ValueError):replay_tree_program(trees(),TokenFacts(facts.codes[0],(3,4)),p,10)
    with pytest.raises(ValueError):model(0,TokenFacts(facts.codes.repeat(2,1),(3,4)),0)
    with pytest.raises(ValueError):model(0,facts,3)
    revised=model(0,TokenFacts(facts.codes.repeat(11,1),(3,4)),10)
    replay_tree_program(trees(),TokenFacts(facts.codes[0],(3,4)),revised,19)


def test_no_rule_or_invalid_code_cannot_silently_select_action():
    with pytest.raises(ValueError):TokenFacts(torch.tensor([3,0]),(3,4)).validate()
    with pytest.raises(ValueError):RuleRevision(torch.tensor([]),0,10).validate(0)


def test_packed_interpreter_matches_individual_proofs_and_critic_is_auxiliary():
    model=TokenProgramReviser((3,4),1,2);program=trees()
    states=torch.tensor([[a,b] for a in range(3) for b in range(4)])
    packed=PackedRuleProgram(program,(3,4));evidence,nodes=packed.evidence(TokenFacts(states,(3,4)))
    for i,state in enumerate(states):
        revision,value=model.generate(0,TokenFacts(state[None],(3,4)),0)
        p,proof,conclusion=replay_tree_program(program,TokenFacts(state,(3,4)),revision,0)
        assert torch.allclose(p,torch.tensor(evidence[i].mean(0),dtype=p.dtype))
        assert torch.equal(conclusion,torch.tensor(evidence[i],dtype=conclusion.dtype))
    before=model(0,TokenFacts(states[:1],(3,4)),0).weights.detach().clone()
    with torch.no_grad():model.critic.weight.fill_(90);model.critic.bias.fill_(-50)
    assert torch.equal(before,model(0,TokenFacts(states[:1],(3,4)),0).weights.detach())


def test_value_feature_detachment_preserves_outputs_and_isolates_value_gradients():
    torch.manual_seed(50)
    ordinary=TokenProgramReviser((3,4),1,2)
    detached=TokenProgramReviser((3,4),1,2,detach_value_features=True)
    with torch.no_grad():
        ordinary.program.weight.normal_(std=.1)
        ordinary.critic.weight.normal_(std=.1)
    detached.load_state_dict(ordinary.state_dict())
    facts=TokenFacts(torch.tensor([[1,2]]),(3,4))
    first,first_value=ordinary.generate(0,facts,0)
    second,second_value=detached.generate(0,facts,0)
    assert torch.equal(first.weights,second.weights)
    assert torch.equal(first_value,second_value)
    (second_value-2).square().backward()
    assert detached.critic.weight.grad.norm()>0
    assert all(p.grad is None for n,p in detached.named_parameters() if not n.startswith('critic.'))
    detached.zero_grad(set_to_none=True)
    revision=detached(0,facts,0)
    probabilities,_,_=replay_tree_program(trees(),TokenFacts(facts.codes[0],(3,4)),revision,0)
    (-probabilities[0].log()).backward()
    assert detached.program.weight.grad.norm()>0
    assert detached.symbols.weight.grad.norm()>0
    assert detached.critic.weight.grad is None
