import pytest
import torch
import torch.nn.functional as F
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton


def example(reset=False):
    torch.manual_seed(56)
    m = ProbabilisticTokenAutomaton(patches=6, vocabulary=8, factors=3, states=4,
                                   width=8, task_banks=True, context_bridge=True,
                                   reset_memory=reset)
    x = torch.randint(8, (2, 6)); tasks = torch.tensor([0, 1])
    p = m.generate(tasks, x[:, None], 0)
    prior = torch.rand(2, 3, 4).softmax(-1); actions = torch.tensor([17, 3])
    return m, x, p, prior, actions, m.step(p, x, prior, actions, 0)


@pytest.mark.parametrize('reset', [False, True])
def test_exact_discrete_sum_matches_independent_enumeration(reset):
    m, x, p, prior, actions, result = example(reset)
    used = F.one_hot(torch.zeros(2, 3, dtype=torch.long), 4).float() if reset else prior
    action = torch.full_like(actions, 17) if reset else actions
    expected = torch.zeros_like(prior)
    for b in range(2):
        for f in range(3):
            o = int(result['observed_ids'][b, f])
            for previous in range(4):
                table = (m.transition[p.tasks[b], f, o, previous, action[b]] + p.transition_delta[b, f, o]).softmax(-1)
                expected[b, f] += used[b, f, previous] * table
    assert torch.allclose(result['state_values'], expected, atol=1e-7, rtol=0)
    manual = torch.zeros(2, 17)
    for b in range(2):
        for r, (i, j) in enumerate(m.pairs):
            for u in range(4):
                for v in range(4):
                    table = (m.conclusions[p.tasks[b], r, u, v] + p.action_delta[b, r, u, v]).softmax(-1)
                    manual[b] += p.rule_weights[b, r] * expected[b, i, u] * expected[b, j, v] * table
    assert torch.allclose(result['probabilities'], manual, atol=2e-7, rtol=0)


def test_entire_multistep_proof_can_be_replayed_without_images():
    m, x, p, prior, actions, result = example()
    for step in range(12):
        if step == 10: p = m.generate(torch.tensor([0, 1]), x[:, None].repeat(1, 11, 1), 10)
        result = m.step(p, x, prior, actions, step)
        proof = m.proof(p, result['observed_ids'], prior, actions, result['state_values'], step)
        assert torch.equal(proof['probabilities'], result['probabilities'])
        prior = proof['state_belief']; actions = proof['probabilities'].argmax(-1)
    with pytest.raises(ValueError): m.proof(p, result['observed_ids'], prior, actions, prior.roll(1, -1), 11)


def test_symbolic_boundaries_reject_features_and_invalid_beliefs():
    m, x, p, prior, actions, result = example()
    with pytest.raises(TypeError): m.step(p, x.float(), prior, actions, 0)
    for bad in [prior * 2, -prior, torch.full_like(prior, float('nan')), prior[:, :, :3]]:
        with pytest.raises(ValueError): m.step(p, x, bad, actions, 0)
    with pytest.raises(ValueError): m.transition_belief(p, prior, prior, actions, 0)
    with pytest.raises(ValueError): m.step(p, x, prior, actions, 10)


def test_action_gradient_reaches_rules_definitions_and_generator():
    m, x, p, prior, actions, result = example()
    loss = -result['probabilities'][:, 5].log().mean()
    loss.backward()
    for param in [m.transition, m.conclusions, m.definitions, m.tokens.weight, m.bridge.weight]:
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm() > 0


def test_auxiliary_erasure_has_no_action_effect_but_rule_erasure_does():
    m, x, p, prior, actions, result = example()
    before = result['probabilities'].detach().clone()
    with torch.no_grad():
        for head in [m.predict, m.inverse, m.reconstruct]:
            for parameter in head.parameters(): parameter.normal_(100, 10)
    assert torch.equal(before, m.step(p, x, prior, actions, 0)['probabilities'])
    with torch.no_grad(): m.conclusions.zero_(); m.conclusion_basis.zero_()
    p = m.generate(torch.tensor([0, 1]), x[:, None], 0)
    after = m.step(p, x, prior, actions, 0)['probabilities']
    assert torch.allclose(after, torch.ones_like(after) / 17, atol=1e-7)
    assert not torch.allclose(before, after)
