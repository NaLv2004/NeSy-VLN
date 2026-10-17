import torch
import pytest
from nesy_vla.sequence_perception import SequencePerception
from nesy_vla.learned_visual_tokens import TrainableAlphabetAutomaton, SymbolTokens


def fixture():
    torch.manual_seed(78)
    block = torch.nn.Conv2d(4, 8, 1)
    vocab = dict(mean=torch.zeros(8), sd=torch.ones(8), axes=torch.eye(8), centers=torch.randn(8, 8))
    frontend = SequencePerception(block, vocab, canonical_batch=4)
    model = TrainableAlphabetAutomaton(patches=49, vocabulary=8, factors=3, states=4,
        width=8, task_banks=True, context_bridge=True)
    features = torch.randn(24, 4, 7, 7)
    return frontend, model, features


def test_sequence_action_training_reaches_visual_block_only_through_hard_tokens():
    frontend, model, features = fixture()
    tokens = frontend(features)
    x = SymbolTokens(tokens.ids.reshape(2, 12, 49), tokens.values.reshape(2, 12, 49, 8))
    assert torch.equal(x.values.detach(), torch.nn.functional.one_hot(x.ids, 8).float())
    task = torch.tensor([0, 1]); _, previous = model.initial_state(2)
    last = torch.full((2,), 17, dtype=torch.long); loss = 0.
    for step in range(12):
        if step % 10 == 0:
            history = x[:, max(0, step-10):step+1]
            program = model.generate(task, history, step)
        result = model.step(program, x[:, step], previous, last, step)
        with torch.no_grad():
            plain_program = model.generate(task, history.ids, program.start)
            plain = model.step(plain_program, x.ids[:, step], previous.detach(), last, step)
            assert torch.equal(plain['probabilities'], result['probabilities'])
            proof = model.proof(program, result['observed_ids'], previous, last, result['state_values'], step)
            assert torch.equal(proof['probabilities'], result['probabilities'])
        loss = loss - result['probabilities'][:, (step+3) % 17].log().mean()
        previous = result['state_values']; last = torch.full((2,), (step+3) % 17, dtype=torch.long)
    loss.backward()
    for parameter in (frontend.block.weight, model.definitions, model.transition, model.conclusions, model.bridge.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    assert all(p.grad is None for p in frontend.alphabet.parameters())


def test_cached_and_single_online_frame_have_identical_hard_ids():
    frontend, _, features = fixture()
    batch = frontend(features).ids
    single = torch.cat([frontend(x[None]).ids for x in features])
    assert torch.equal(batch, single)


def test_forward_interpreter_rejects_continuous_features_and_soft_surrogate_values():
    frontend, model, features = fixture()
    tokens = frontend(features[:2]); task = torch.tensor([0, 1])
    with pytest.raises(ValueError): model.generate(task, features[:2, None], 0)
    with pytest.raises(TypeError): model.generate(task, tokens.ids[:, None].float(), 0)
    with pytest.raises(ValueError):
        model.generate(task, SymbolTokens(tokens.ids[:, None], torch.ones_like(tokens.values[:, None])/8), 0)


def test_erasing_rules_removes_visual_control_after_sequence_frontend():
    frontend, model, features = fixture()
    tokens = frontend(features[:2]); task = torch.tensor([0, 1]); _, state = model.initial_state(2)
    with torch.no_grad(): model.conclusions.zero_(); model.conclusion_basis.zero_()
    program = model.generate(task, tokens[:, None], 0)
    result = model.step(program, tokens, state, torch.full((2,), 17, dtype=torch.long), 0)
    assert torch.allclose(result['probabilities'], torch.ones_like(result['probabilities'])/17, atol=1e-7)
