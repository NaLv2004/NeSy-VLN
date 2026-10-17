import pytest,torch
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.observation_inverse import observation_equalities,observation_inverse_logits


def example():
    torch.manual_seed(76)
    model=ProbabilisticTokenAutomaton(patches=6,vocabulary=8,factors=3,states=4,width=8,task_banks=True,context_bridge=True)
    before=torch.randint(8,(8,6));after=torch.randint(8,(8,6));tasks=torch.arange(8)%2
    program=model.generate(tasks,torch.stack([before,after],1),10)
    return model,program,before,after


def test_auxiliary_cannot_read_transition_action_memory_or_action_conclusions():
    model,program,before,after=example()
    baseline=observation_inverse_logits(model,program,before,after,10)
    with torch.no_grad():
        model.transition.normal_(0,30);model.conclusions.normal_(0,30)
        program.transition_delta.normal_(0,30);program.action_delta.normal_(0,30)
    changed=observation_inverse_logits(model,program,before,after,10)
    assert torch.equal(baseline,changed)


def test_inverse_supervision_trains_grounding_but_not_action_bearing_transition():
    model,program,before,after=example()
    logits=observation_inverse_logits(model,program,before,after,10)
    torch.nn.functional.cross_entropy(logits,torch.arange(8)%17).backward()
    assert model.observation[0].weight.grad.norm()>0
    assert model.definitions.grad.norm()>0
    assert model.transition.grad is None and model.conclusions.grad is None


def test_grounding_matches_forward_predicates_at_revision_without_recurrent_input():
    model,program,before,after=example();_,prior=model.initial_state(8)
    result=model.step(program,after,prior,torch.arange(8),10)
    grounded=observation_equalities(model,program,after,10)
    assert torch.equal(grounded,result['observed_values'])
    assert torch.equal(grounded.detach(),grounded.detach().round())
    logits=observation_inverse_logits(model,program,before,after,10)
    manual=model.inverse_action(observation_equalities(model,program,before,10),grounded)
    assert torch.equal(logits,manual)


def test_continuous_visual_inputs_are_rejected():
    model,program,before,after=example()
    with pytest.raises(TypeError):observation_inverse_logits(model,program,before.float(),after,10)
