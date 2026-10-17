import pytest
import torch
from nesy_vla.symbols import tokenize,replay_observed_tokens


def test_observed_token_replay_preserves_recorded_forward_and_grounding_gradient():
    fresh_values=torch.tensor([.501,.499,.7],requires_grad=True)
    replay=replay_observed_tokens(tokenize(fresh_values),[0,1,1])
    assert torch.equal(replay.codes,torch.tensor([0,1,1]))
    assert torch.equal(replay.values,torch.tensor([0.,1.,1.]))
    (replay.values*torch.tensor([2.,3.,5.])).sum().backward()
    assert torch.equal(fresh_values.grad,torch.tensor([2.,3.,5.]))


def test_replay_rejects_continuous_or_malformed_tokens():
    state=tokenize(torch.tensor([.2,.7]))
    for codes in [[0.,1.],[.2,.7],[1]]:
        with pytest.raises(TypeError):replay_observed_tokens(state,codes)
    with pytest.raises(ValueError):replay_observed_tokens(state,[0,2])


def test_replayed_symbolic_likelihood_survives_encoder_threshold_rounding():
    from nesy_vla.neural_logic import NeuralSymbolicLogic
    from nesy_vla.generator import ProgramDecoder
    torch.manual_seed(88)
    p=ProgramDecoder(hidden=8,k=4,d=16,r=24,a=3,structured=True)(torch.randn(1,5,8))
    logic=NeuralSymbolicLogic(4,3,24)
    original=tokenize(torch.tensor([.4999,.5001,.2,.8]))
    rounded=tokenize(torch.tensor([.5001,.4999,.2,.8]))
    replay=replay_observed_tokens(rounded,original.codes)
    assert torch.equal(logic.infer(p,original)['probs'],logic.infer(p,replay)['probs'])
