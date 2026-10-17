import json
import torch
import pytest
from nesy_vla.generator import ProgramDecoder,program_distance
from nesy_vla.program import Program
from nesy_vla.neural_logic import NeuralSymbolicLogic


def test_multieffect_program_roundtrip_and_exclusive_polarities():
    torch.manual_seed(13)
    head=ProgramDecoder(hidden=32,k=8,d=32,r=48,structured=True,multi_effects=True)
    p=head(torch.randn(1,6,32));p.validate()
    assert p.effect_mode=='per_predicate'
    add,delete=p.effects.chunk(2,-1)
    assert torch.all(add+delete<=1.)
    assert torch.all(p.effects.sum(-1)>1.)
    restored=Program.from_record(json.loads(json.dumps(p.record())))
    assert restored.effect_mode==p.effect_mode and torch.equal(restored.effects,p.effects)
    p.harden().validate()
    assert torch.all(p.harden().effects[:,:8]*p.harden().effects[:,8:]==0)
    q=head(torch.randn(1,6,32));loss=program_distance(q,p)
    assert torch.isfinite(loss);loss.backward()
    assert head.effect.weight.grad.norm()>0


def test_generated_effect_set_can_change_several_discrete_predicates():
    torch.manual_seed(13)
    head=ProgramDecoder(hidden=32,k=8,d=32,r=48,structured=True,multi_effects=True)
    p=head(torch.randn(1,6,32)).detach();logic=NeuralSymbolicLogic(8,17,48)
    p.weights.zero_();r=logic.effect_slice.start;p.weights[r]=1.
    p.premises[r].zero_();p.premises[r,:,0]=1.
    p.action_conditions[r].zero_();p.action_conditions[r,4]=1.
    p.effects.zero_();p.effects[r,[0,2,4]]=1.;p.effects[r,8+6]=1.
    p.validate()
    source=torch.zeros(8);source[6]=1.
    expected=source.clone();expected[[0,2,4]]=1.;expected[6]=0.
    predicted,_,_=logic.transition(p,source,torch.ones(8),4,torch.ones(8))
    assert torch.equal((predicted>=.5).float(),expected)
    # Removing the generating rule removes every change, not just a displayed explanation.
    p.weights.zero_()
    absent,_,_=logic.transition(p,source,torch.ones(8),4,torch.ones(8))
    assert torch.equal(absent,source)


def test_multieffect_batched_transitions_equal_enumerated_actions():
    torch.manual_seed(17)
    head=ProgramDecoder(hidden=32,k=8,d=32,r=48,structured=True,multi_effects=True)
    p=head(torch.randn(1,6,32));logic=NeuralSymbolicLogic(8,17,48)
    source=torch.randint(2,(8,)).float();keep=torch.rand(17,8)
    all_predictions,_,_=logic.transition_all(p,source,torch.ones(8),keep)
    for action in range(17):
        expected,_,_=logic.transition(p,source,torch.ones(8),action,keep[action])
        assert torch.allclose(all_predictions[action],expected,atol=1e-6)
    invalid=p.map(lambda x:x);invalid.effects=torch.ones_like(p.effects)
    with pytest.raises(ValueError):invalid.validate()
