"""Exact symbolic planning restricted by training-observed applicability.

This module sees only anonymous integer states, opaque action IDs, an immutable
neural transition table and a boolean applicability mask. It has no simulator,
images, physical-state decoder, action-name semantics or neural-network calls.
"""
import hashlib
import numpy as np


def plan_legal(program, initial, applicable, max_horizon=10):
    program.validate()
    initial = np.asarray(initial, dtype=np.int64)
    table = program.transitions
    successor = table.argmax(-1)
    states, actions = successor.shape
    if initial.shape != (1,) or not 1 <= max_horizon <= 10:
        raise ValueError('Invalid planning request')
    applicable = np.asarray(applicable, dtype=bool).copy()
    if applicable.shape != (states, actions):
        raise ValueError('Applicability shape mismatch')
    goal = int(program.subgoal[0])
    # STOP cannot be used as a fictitious transition to another state. Keep
    # the network's own successor; permit termination only at its exact goal.
    applicable[:, 0] &= (np.arange(states) == goal) & (successor[:, 0] == np.arange(states))
    log_confidence = np.rint(np.log(table.max(-1).astype(np.float64).clip(1e-30)) * 10000).astype(np.int64)
    impossible = -10**15
    exact = np.full(states, impossible, dtype=np.int64)
    exact[goal] = 0
    base = dict(subgoal=program.subgoal.tolist(), exhaustive_finite_state_search=True,
                program_sha256=program.digest(), applicability_sha256=hashlib.sha256(applicable.tobytes()).hexdigest(),
                all_actions_applicability_checked=True)
    expanded = 0
    for horizon in range(1, max_horizon + 1):
        confidence = log_confidence * (horizon + 1) - 1
        support = exact.copy()
        policies = [None]
        for depth in range(1, horizon + 1):
            derivation = confidence + support[successor]
            derivation[:, 0] = confidence[:, 0] + exact[successor[:, 0]]
            derivation[~applicable] = impossible
            policy = derivation.argmax(-1)
            policies.append(policy)
            support = derivation[np.arange(states), policy]
            expanded += states * actions
        if support[int(initial[0])] < impossible // 2:
            continue
        state = int(initial[0])
        trajectory = [[state]]
        chosen = []
        for remaining in range(horizon, 0, -1):
            action = int(policies[remaining][state])
            assert applicable[state, action]
            chosen.append(action)
            state = int(successor[state, action])
            trajectory.append([state])
            if action == 0:
                break
        assert state == goal
        return dict(**base, found=True, actions=chosen, states=trajectory,
                    planning_horizon=horizon, expanded=expanded, goal_distance=0)
    return dict(**base, found=False, actions=[], states=[initial.tolist()],
                planning_horizon=max_horizon, expanded=expanded, goal_distance=None)
