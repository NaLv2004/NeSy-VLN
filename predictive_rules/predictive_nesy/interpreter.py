"""Standalone NumPy rule interpreter and explicit finite-horizon planner.

Importing this module does not import torch. The exported program is sufficient
to reproduce every candidate transition, chosen plan and executed action.
"""
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class Program:
    parents: np.ndarray
    transitions: np.ndarray
    risks: np.ndarray
    goals: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    definitions: np.ndarray
    start: int
    interval: int = 10

    def validate(self):
        f, a, k, k2, k3 = self.transitions.shape
        assert k == k2 == k3 and self.parents.shape == (f, 2)
        assert self.parents.dtype.kind in "iu" and np.all((self.parents >= 0) & (self.parents < f))
        assert self.risks.shape == (f, a, k, k)
        p = f * (f - 1) // 2
        assert self.goals.shape == self.values.shape == (p, k, k)
        assert self.weights.shape == (p,) and abs(float(self.weights.sum()) - 1) < 1e-5
        assert self.definitions.shape[:2] == (f, k)
        for value in (self.transitions, self.risks, self.goals, self.values, self.weights, self.definitions):
            assert np.isfinite(value).all()
        assert np.allclose(self.transitions.sum(-1), 1, atol=1e-5)
        for value in (self.transitions, self.risks, self.goals, self.values, self.weights):
            assert np.all((value >= 0) & (value <= 1))
        assert self.start >= 0 and self.start % self.interval == 0
        return self

    def digest(self):
        h = hashlib.sha256()
        for name in ("parents", "transitions", "risks", "goals", "values", "weights", "definitions"):
            h.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        h.update(str((self.start, self.interval)).encode())
        return h.hexdigest()

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            kwargs = {name: data[name] for name in data.files}
        kwargs["start"] = int(kwargs["start"])
        kwargs["interval"] = int(kwargs["interval"])
        return cls(**kwargs).validate()


def export_program(generated, index=0, interval=10):
    arrays = {name: generated[name][index].detach().float().cpu().numpy().copy()
              for name in ("transitions", "risks", "goals", "values", "weights", "definitions")}
    arrays["parents"] = generated["parents"][index].detach().argmax(-1).cpu().numpy().astype(np.int64)
    return Program(**arrays, start=generated["start"], interval=interval).validate()


def transition(program, states, actions):
    states = np.asarray(states, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    f = np.arange(states.shape[-1])
    u = states[..., program.parents[:, 0]]
    v = states[..., program.parents[:, 1]]
    probabilities = program.transitions[f, actions[..., None], u, v]
    return probabilities.argmax(-1), probabilities


def scalar(program, states, name):
    states = np.asarray(states, dtype=np.int64)
    values = getattr(program, name)
    result = np.zeros(states.shape[:-1], dtype=np.float64)
    for r, (i, j) in enumerate(combinations(range(states.shape[-1]), 2)):
        result += program.weights[r] * values[r, states[..., i], states[..., j]]
    return result


def risk(program, states, actions):
    f = np.arange(states.shape[-1])
    u = states[..., program.parents[:, 0]]
    v = states[..., program.parents[:, 1]]
    return program.risks[f, np.asarray(actions)[..., None], u, v].mean(-1)


def plan(program, initial, horizon=10, beam_width=32, gamma=.97,
         step_cost=.02, collision_cost=.2):
    """Search all action successors, retaining complete candidate sequences.

    STOP terminates a branch. Nonterminal leaves use the generated scalar value
    rules. No candidate executes in the real environment or in a simulator.
    """
    program.validate()
    if not 1 <= horizon <= program.interval:
        raise ValueError("A plan cannot cross a rule-revision boundary")
    states = np.asarray(initial, dtype=np.int64)[None]
    f, a, k, _, _ = program.transitions.shape
    if states.shape != (1, f) or not np.all((states >= 0) & (states < k)):
        raise ValueError("Invalid symbolic initial state")
    paths = np.zeros((1, 0), dtype=np.int64)
    cost = np.zeros(1)
    terminal = np.zeros(1, dtype=bool)
    trajectory_states = states[:, None].copy()
    expanded = 0
    for depth in range(horizon):
        parent = np.repeat(np.arange(len(states)), a)
        actions = np.tile(np.arange(a), len(states))
        alive = ~terminal[parent]
        # Keep one continuation of already completed plans.
        valid = alive | (actions == 0)
        parent, actions, alive = parent[valid], actions[valid], alive[valid]
        before = states[parent]
        after, _ = transition(program, before, actions)
        stopped = (actions == 0) | ~alive
        after = np.where(stopped[:, None], before, after)
        rewards = -step_cost - collision_cost * risk(program, before, actions)
        rewards = np.where(actions == 0, 2 * scalar(program, before, "goals") - 1, rewards)
        rewards = np.where(alive, rewards, 0)
        new_cost = cost[parent] + gamma ** depth * rewards
        score = new_cost + np.where(stopped, 0, gamma ** (depth + 1) * scalar(program, after, "values"))
        expanded += len(parent)
        # Stable sorting makes tie handling and independent replay exact.
        kept = np.argsort(-score, kind="stable")[:beam_width]
        states = after[kept]
        cost = new_cost[kept]
        terminal = stopped[kept]
        paths = np.concatenate([paths[parent[kept]], actions[kept, None]], -1)
        trajectory_states = np.concatenate([trajectory_states[parent[kept]], states[:, None]], 1)
    selected = paths[0].tolist()
    if 0 in selected:
        selected = selected[:selected.index(0) + 1]
    return dict(actions=selected, states=trajectory_states[0, :len(selected) + 1].tolist(),
                score=float(score[kept[0]]), expanded=expanded,
                program_sha256=program.digest(), planning_horizon=horizon,
                beam_width=beam_width)


class BlockController:
    """No observation/feedback argument exists in the action execution API."""
    def __init__(self, interval=10):
        self.interval = interval
        self.step = 0
        self.program = None
        self.current = None
        self.queue = []
        self.finished = False
        self.program_hash = None

    def revise(self, program, state, *, horizon=None, beam_width=32):
        if self.finished or self.step % self.interval or self.queue:
            raise ValueError("Rule/state correction outside a revision boundary")
        if program.start != self.step or program.interval != self.interval:
            raise ValueError("Program boundary mismatch")
        self.program = program.validate()
        self.program_hash = program.digest()
        self.current = np.asarray(state, dtype=np.int64).copy()
        result = plan(program, self.current, horizon=horizon or self.interval, beam_width=beam_width)
        self.queue = result["actions"].copy()
        return result

    def next_action(self):
        if self.finished or not self.queue:
            raise ValueError("A boundary revision or episode end is required")
        action = self.queue.pop(0)
        before = self.current.copy()
        if action != 0:
            self.current, probabilities = transition(self.program, self.current, np.asarray(action))
        else:
            probabilities = np.eye(self.program.transitions.shape[-1])[self.current]
            self.finished = True
            self.queue.clear()
        proof = dict(step=self.step, program_start=self.program.start, before=before.tolist(),
                     action=action, after=self.current.tolist(), transition_probabilities=probabilities.tolist(),
                     program_sha256=self.program_hash)
        self.step += 1
        return action, proof
