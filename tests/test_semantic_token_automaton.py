import pytest
import torch
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.semantic_token_automaton import SemanticTokens,SemanticTokenAutomaton


def example():
    torch.manual_seed(67)
    args=dict(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True)
    base=ProbabilisticTokenAutomaton(**args);model=SemanticTokenAutomaton(torch.randn(64,32),**args)
    model.initialize_common(base.state_dict())
    tokens=SemanticTokens(torch.randint(8,(2,6)),torch.randint(64,(2,12)),torch.ones(2,12,dtype=torch.bool))
    goal=torch.tensor([0,1]);_,memory=model.initial_state(2);action=torch.full((2,),17)
    return base,model,tokens,goal,memory,action


def test_initial_symbolic_program_matches_original_exactly():
    base,m,x,goal,memory,a=example()
    first=base.step(base.generate(goal,x.visual[:,None],0),x.visual,memory,a,0)
    second=m.step(m.generate(goal,x[:,None],0),x,memory,a,0)
    assert torch.equal(first['probabilities'],second['probabilities'])


def test_changed_text_can_change_generated_program_only_through_rule_inference():
    _,m,x,goal,memory,a=example()
    with torch.no_grad():m.text_to_tokens.weight.normal_(0,.1)
    p=m.generate(goal,x[:,None],0);r=m.step(p,x,memory,a,0)
    y=SemanticTokens(x.visual,torch.zeros_like(x.text),x.mask)
    alternative=m.step(m.generate(goal,y[:,None],0),y,memory,a,0)
    assert not torch.equal(r['probabilities'],alternative['probabilities'])
    proof=m.proof(p,r['observed_ids'],memory,a,r['state_values'],0)
    assert torch.equal(proof['probabilities'],r['probabilities'])
    with torch.no_grad():m.conclusions.zero_();m.conclusion_basis.zero_()
    p=m.generate(goal,x[:,None],0);r=m.step(p,x,memory,a,0)
    assert torch.allclose(r['probabilities'],torch.ones_like(r['probabilities'])/17,atol=1e-7)


def test_rejects_image_features_hidden_states_and_noninteger_text():
    _,m,x,goal,memory,a=example()
    with pytest.raises(TypeError):m.step(m.generate(goal,x[:,None],0),x.visual.float(),memory,a,0)
    with pytest.raises(TypeError):m.generate(goal,SemanticTokens(x.visual[:,None],x.text[:,None].float(),x.mask[:,None]),0)
    with pytest.raises(ValueError):m.generate(goal,SemanticTokens(x.visual[:,None],x.text[:,None],x.mask[:,None,:3]),0)


def test_caption_gradient_and_pretrained_embedding_checkpoint_contract():
    _,m,x,goal,memory,a=example()
    with torch.no_grad():m.text_to_tokens.weight.normal_(0,.1)
    p=m.generate(goal,x[:,None],0);r=m.step(p,x,memory,a,0);(-r['probabilities'][:,5].log().mean()).backward()
    for parameter in [m.text_project.weight,m.text_reader.weight_ih_l0,m.text_to_tokens.weight,m.definitions,m.conclusions]:
        assert parameter.grad is not None and parameter.grad.norm()>0
    assert not m.word_table.requires_grad and 'word_table' not in m.state_dict()


def test_absent_text_leaves_original_tokens_even_after_text_bias_changes():
    base,m,x,goal,memory,a=example()
    with torch.no_grad():m.text_to_tokens.weight.normal_();m.text_to_tokens.bias.normal_()
    absent=SemanticTokens(x.visual,x.text,torch.zeros_like(x.mask))
    assert torch.equal(m.embed(absent),base.embed(x.visual))
