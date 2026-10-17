"""Compare the symbolic DP against independent 90-digit Decimal arithmetic."""
from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from planner import Program, plan


def decimal_reference(t, p, goal, observations, horizon):
    with localcontext() as context:
        context.prec = 90
        n, actions = t.shape
        probabilities = []
        for row in p:
            choices = []
            for branch in row:
                numbers = [Decimal(float(x)) for x in branch]
                mass = sum(numbers)
                choices.append([x / mass for x in numbers])
            probabilities.append(choices)
        value = [Decimal(0)] * n
        results = []
        gm, go = divmod(goal, observations)
        for _ in range(horizon):
            all_q = []
            for state in range(n):
                q = []
                for action in range(actions):
                    nm = int(t[state, action])
                    row = probabilities[state][action]
                    q.append(row[go] if nm == gm else Decimal(0)) if action == 0 else q.append(
                        sum(row[o] * value[nm * observations + o] for o in range(observations)))
                all_q.append(q)
            value = [max(q) for q in all_q]
            results.append((value, all_q))
        return results


def verify(t, p, goal, observations, horizon=32):
    candidate = Program(t, p, goal, observations=observations).validate()
    raw = p.tobytes()
    ref = decimal_reference(t, p, goal, observations, horizon)
    candidate.extend(horizon)
    for h, (values, choices) in enumerate(ref, 1):
        for state, optimum in enumerate(values):
            action = candidate._policies[h][state]
            # Do not use float-rounded success to judge action optimality.
            with localcontext() as context:
                context.prec = 90
                # Float64 cannot rank differences below its relative precision.
                # Scale tolerance by the smaller tail, never by success~1.
                tolerance=Decimal('2e-14')*min(optimum,1-optimum)+Decimal('1e-80')
                assert optimum - choices[state][action] <= tolerance, (h,state,action)
                assert np.isclose(candidate._levels[h][state], float(optimum), atol=1e-14)
                assert np.isclose(candidate._failures[h][state], float(1-optimum), rtol=1e-12, atol=1e-80)
    assert p.tobytes() == raw
    return candidate


def main():
    t = np.zeros((2, 17), np.int64)
    p = np.zeros((2, 17, 2), np.float32)
    p[0, :, 0] = 1; p[1, :, 1] = 1
    p[0, 1] = [.4, .6]; p[0, 2] = [.2, .8]
    candidate = verify(t, p, 1, 2)
    proof = plan(candidate, 0, 0, 32)
    assert proof['actions'][0] == 2 and proof['actions'][-1] == 0
    assert 0 < proof['model_failure_probability'] < 1e-20
    spec = importlib.util.spec_from_file_location('legacy', Path(__file__).parents[1]/'contextual_symbols_v48/planner.py')
    legacy = importlib.util.module_from_spec(spec); sys.modules[spec.name] = legacy; spec.loader.exec_module(legacy)
    old = legacy.plan(legacy.Program(t, p, 1, observations=2), 0, 0, 32)
    assert old['actions'][0] != 2, old
    assert old['model_goal_probability'] > 1
    rng = np.random.default_rng(501017)
    for _ in range(4):
        transitions = rng.integers(0, 2, size=(8,17))
        probabilities = rng.dirichlet(np.ones(4), size=(8,17)).astype(np.float32)
        verify(transitions, probabilities, int(rng.integers(0,8)), 4)
    dead = np.zeros((2,17,2),np.float32); dead[:,:,0] = 1
    assert not plan(Program(t,dead,1,observations=2),0,0,32)['found']
    report = dict(passed=True,decimal_precision=90,verified_state_horizon_cases=1088,
        legacy_fixture_action=old['actions'][0],correct_action=proof['actions'][0],
        legacy_invalid_probability=old['model_goal_probability'],
        stable_failure_probability=proof['model_failure_probability'],
        raw_table_bytes_unchanged=True,dead_end_does_not_fallback=True)
    (Path(__file__).parent/'arithmetic_contract.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__': main()
