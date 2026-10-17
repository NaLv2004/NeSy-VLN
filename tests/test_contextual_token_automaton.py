import pytest,torch
from nesy_vla.contextual_token_automaton import FrozenCaptionLookup,ContextualTokenAutomaton
from nesy_vla.semantic_token_automaton import SemanticTokens
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton

def example():
    torch.manual_seed(71)
    args=dict(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True)
    base=ProbabilisticTokenAutomaton(**args)
    encoder=FrozenCaptionLookup([(1,2),(2,3)],torch.randn(2,32))
    model=ContextualTokenAutomaton(encoder,**args);model.initialize_common(base.state_dict())
    x=SemanticTokens(torch.randint(8,(2,6)),torch.tensor([[1,2,0],[2,3,0]]),torch.tensor([[True,True,False]]*2))
    task=torch.tensor([0,1]);_,prior=model.initial_state(2);action=torch.full((2,),17)
    return base,model,x,task,prior,action

def test_cache_uses_full_discrete_sequence_not_images_or_hidden_state_input():
    _,m,x,_,_,_=example();e=m.text_encoder
    assert torch.equal(e(x.text,x.mask),e.representations[:2].float())
    with pytest.raises(TypeError):e(x.text.float(),x.mask)
    with pytest.raises(KeyError):e(torch.tensor([[1,3,0]]),x.mask[:1])
    assert not e.state_dict() and not e.representations.requires_grad

def test_initial_and_no_text_programs_preserve_common_model():
    base,m,x,task,prior,a=example()
    p=m.generate(task,x[:,None],0);r=m.step(p,x,prior,a,0)
    original=base.step(base.generate(task,x.visual[:,None],0),x.visual,prior,a,0)
    assert torch.equal(r['probabilities'],original['probabilities'])
    with torch.no_grad():m.text_to_tokens.weight.normal_();m.text_to_tokens.bias.normal_()
    empty=SemanticTokens(x.visual,x.text,torch.zeros_like(x.mask))
    assert torch.equal(m.embed(empty),base.embed(x.visual))

def test_context_changes_states_and_rules_with_exact_action_proof():
    _,m,x,task,prior,a=example()
    with torch.no_grad():m.text_to_tokens.weight.normal_(0,.1)
    p=m.generate(task,x[:,None],0);r=m.step(p,x,prior,a,0)
    other=SemanticTokens(x.visual,x.text.flip(0),x.mask)
    changed=m.step(m.generate(task,other[:,None],0),other,prior,a,0)
    assert not torch.equal(changed['probabilities'],r['probabilities'])
    proof=m.proof(p,r['observed_ids'],prior,a,r['state_values'],0)
    assert torch.equal(proof['probabilities'],r['probabilities'])
    (-r['probabilities'][:,6].log().mean()).backward()
    for parameter in [m.text_to_tokens.weight,m.definitions,m.conclusions]:assert parameter.grad.norm()>0
    with torch.no_grad():m.conclusions.zero_();m.conclusion_basis.zero_()
    erased=m.step(m.generate(task,x[:,None],0),x,prior,a,0)
    assert torch.allclose(erased['probabilities'],torch.ones_like(erased['probabilities'])/17,atol=1e-7)
