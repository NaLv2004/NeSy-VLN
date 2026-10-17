import pytest
import torch
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.automaton_policy_gradient import SymbolicValue, replay, advantages, policy_objective, configure_trainable


def case():
    torch.manual_seed(62)
    m = ProbabilisticTokenAutomaton(patches=6, vocabulary=8, factors=3, states=4, width=8, task_banks=True, context_bridge=True)
    return m, torch.randint(8, (8, 12, 6)), torch.arange(8) % 2, torch.randint(17, (8, 12))


def test_replay_matches_online_across_program_revision_and_has_exact_proof():
    m, x, task, a = case(); _, memory = m.initial_state(8); previous = torch.full((8,), 17)
    expected = []
    with torch.no_grad():
        for t in range(12):
            if t % 10 == 0: program = m.generate(task, x[:, max(0, t-10):t+1], t)
            r = m.step(program, x[:, t], memory, previous, t)
            proof = m.proof(program, r['observed_ids'], memory, previous, r['state_values'], t)
            assert torch.equal(proof['probabilities'], r['probabilities'])
            expected.append(r['probabilities']); memory = r['state_values']; previous = a[:, t]
    result = replay(m, x, task, a)
    assert torch.equal(torch.stack(expected, 1), result['probabilities'])
    p = result['probabilities']; old = p.detach().log().gather(-1, a[..., None]).squeeze(-1)
    loss, audit = policy_objective(p, a, old, torch.randn(8, 12), torch.ones(8, 12, dtype=torch.bool))
    assert audit['max_ratio_error'] == 0
    loss.backward()
    for parameter in [m.transition, m.conclusions, m.definitions, m.tokens.weight, m.bridge.weight]:
        assert parameter.grad is not None and parameter.grad.norm() > 0


def test_value_loss_cannot_change_actor_or_action_selection():
    m, x, task, a = case(); result = replay(m, x, task, a); before = result['probabilities'].detach()
    critic = SymbolicValue(m.factors, m.states, m.tasks)
    critic(result['beliefs'][:, 0], task).square().mean().backward()
    assert all(p.grad is None for p in m.parameters())
    with torch.no_grad():
        for p in critic.parameters(): p.normal_(100, 10)
    assert torch.equal(replay(m, x, task, a)['probabilities'], before)


def test_gae_does_not_leak_rewards_past_terminal():
    reward = torch.tensor([[1., 2., 100.], [1., 2., 3.]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    adv, returns = advantages(reward, torch.zeros_like(reward), valid, 1., 1.)
    assert torch.equal(adv, torch.tensor([[3., 2., 0.], [6., 5., 3.]]))
    assert torch.equal(adv, returns)
    with pytest.raises(ValueError): advantages(reward, reward, torch.tensor([[True, False, True], [True, True, True]]))


def test_symbolic_input_boundary_and_matched_frozen_control():
    m, x, task, a = case()
    with pytest.raises(TypeError): replay(m, x.float(), task, a)
    with pytest.raises(ValueError): replay(m, x[:4], task[:4], a[:4])
    configure_trainable(m, 'action_rules_only')
    assert {n for n, p in m.named_parameters() if p.requires_grad} == {'conclusions', 'conclusion_basis', 'weight_logits', 'weight_basis'}
    with pytest.raises(TypeError): replay(m, x, task, a.float())
