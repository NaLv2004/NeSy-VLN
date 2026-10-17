"""Neural modules are legal only at a rule-revision boundary.

All transition, goal, value, risk, parent-selection and definition parameters
are emitted by the hypernetwork. There is no base rule bank, low-rank rule
delta, per-action policy head, recurrent neural transition, or map input.
"""
from dataclasses import dataclass, asdict
from itertools import combinations
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    patches: int = 49
    vocabulary: int = 64
    tasks: int = 2
    factors: int = 4
    states: int = 16
    actions: int = 17
    width: int = 64
    hidden: int = 128
    interval: int = 10
    history: int = 11


def straight_through(probabilities):
    discrete = F.one_hot(probabilities.argmax(-1), probabilities.shape[-1]).to(probabilities)
    return discrete + probabilities - probabilities.detach()


class RowDecoder(nn.Module):
    """Every row is decoded in full; query IDs have no assigned semantics."""
    def __init__(self, sizes, hidden, outputs):
        super().__init__()
        self.queries = nn.ModuleList(nn.Embedding(n, hidden) for n in sizes)
        self.context = nn.Linear(hidden, hidden)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, outputs))
        indices = torch.cartesian_prod(*(torch.arange(n) for n in sizes))
        if len(sizes) == 1:
            indices = indices[:, None]
        self.register_buffer("indices", indices)
        self.sizes = tuple(sizes)
        self.outputs = outputs

    def forward(self, context):
        query = sum(table(self.indices[:, j]) for j, table in enumerate(self.queries))
        rows = self.decoder(F.gelu(query[None] + self.context(context)[:, None]))
        return rows.reshape(len(context), *self.sizes, self.outputs)


class GeneratedWorldModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or ModelConfig()
        c = self.cfg
        self.pairs = list(combinations(range(c.factors), 2))
        self.tokens = nn.Embedding(c.patches * c.vocabulary, 8)
        self.register_buffer("offsets", torch.arange(c.patches) * c.vocabulary)
        self.observation = nn.Sequential(nn.Linear(c.patches * 8, 256), nn.GELU(),
            nn.Linear(256, c.factors * c.width), nn.LayerNorm(c.factors * c.width))
        self.context_read = nn.Sequential(nn.Linear(c.patches * 8, c.hidden), nn.GELU(), nn.LayerNorm(c.hidden))
        self.action_embed = nn.Embedding(c.actions + 1, c.hidden)
        self.task = nn.Embedding(c.tasks, c.hidden)
        self.history = nn.GRU(c.hidden, c.hidden, batch_first=True)
        self.context_norm = nn.LayerNorm(c.hidden)
        self.definitions = RowDecoder([c.factors, c.states], c.hidden, c.width)
        # Both parent-factor identities are generated, not hand-picked.
        self.parents = RowDecoder([c.factors, 2], c.hidden, c.factors)
        self.transitions = RowDecoder([c.factors, c.actions, c.states, c.states], c.hidden, c.states)
        self.risks = RowDecoder([c.factors, c.actions, c.states, c.states], c.hidden, 1)
        self.goals = RowDecoder([len(self.pairs), c.states, c.states], c.hidden, 1)
        self.values = RowDecoder([len(self.pairs), c.states, c.states], c.hidden, 1)
        self.rule_weights = nn.Linear(c.hidden, len(self.pairs))
        # Training-only heads. These cannot be exported to the interpreter.
        self.reconstruct = nn.Linear(c.factors * c.states, c.patches * c.vocabulary)
        self.inverse = nn.Sequential(nn.Linear(2 * c.factors * c.states, 128), nn.GELU(), nn.Linear(128, c.actions))

    def embed(self, codes):
        if codes.dtype != torch.long or codes.shape[-1] != self.cfg.patches:
            raise ValueError("Expected integer visual patch codes")
        return self.tokens(codes + self.offsets).flatten(-2)

    def generate(self, tasks, history, previous_actions, lengths, *, start):
        if start < 0 or start % self.cfg.interval:
            raise ValueError("Perception and rule generation are permitted only at revision boundaries")
        if history.ndim != 3 or history.shape[1] > min(start + 1, self.cfg.history):
            raise ValueError("History must be causal and end at this revision boundary")
        if previous_actions.shape != history.shape[:2]:
            raise ValueError("Each observed frame must have its preceding executed-action ID")
        inputs = self.context_read(self.embed(history)) + self.action_embed(previous_actions)
        hidden, _ = self.history(inputs, self.task(tasks)[None])
        context = self.context_norm(hidden[torch.arange(len(tasks), device=tasks.device), lengths - 1])
        return dict(
            definitions=self.definitions(context),
            parents=self.parents(context).float().softmax(-1),
            transitions=self.transitions(context).float().softmax(-1),
            risks=self.risks(context).float().squeeze(-1).sigmoid(),
            goals=self.goals(context).float().squeeze(-1).sigmoid(),
            values=self.values(context).float().squeeze(-1).sigmoid(),
            weights=self.rule_weights(context).float().softmax(-1),
            start=start,
        )

    def ground(self, program, codes):
        """Boundary perception, or training targets ONLY; never a runtime step."""
        shape = codes.shape[:-1]
        features = self.observation(self.embed(codes)).reshape(*shape, self.cfg.factors, self.cfg.width)
        if codes.ndim == 3:
            logits = torch.einsum("btfd,bfkd->btfk", F.normalize(features, dim=-1), F.normalize(program["definitions"], dim=-1))
        else:
            logits = torch.einsum("bfd,bfkd->bfk", F.normalize(features, dim=-1), F.normalize(program["definitions"], dim=-1))
        return (5. * logits.float()).softmax(-1)

    def export_config(self):
        return asdict(self.cfg)


def parent_beliefs(program, state):
    selectors = straight_through(program["parents"])
    return torch.einsum("bfpg,bgk->bfpk", selectors, state)


def transition_all(program, state):
    """Pure table contraction. No model, observation or latent feature input."""
    parents = parent_beliefs(program, state)
    return torch.einsum("bfu,bfv,bfauvk->bafk", parents[:, :, 0], parents[:, :, 1], program["transitions"])


def advance(program, state, action, *, hard=True):
    result = transition_all(program, state)[torch.arange(len(state), device=state.device), action]
    return straight_through(result) if hard else result


def read_scalar(program, state, name, pairs):
    values = []
    for index, (i, j) in enumerate(pairs):
        values.append(torch.einsum("bu,bv,buv->b", state[:, i], state[:, j], program[name][:, index]))
    return (torch.stack(values, -1) * program["weights"]).sum(-1)


def risk_all(program, state):
    parents = parent_beliefs(program, state)
    return torch.einsum("bfu,bfv,bfauv->ba", parents[:, :, 0], parents[:, :, 1], program["risks"]) / state.shape[1]


def one_step_scores(program, state, pairs, gamma=.97, step_cost=.02, collision_cost=.2):
    """Bellman scores from generated transition/outcome tables, not an actor."""
    future = transition_all(program, state)
    # The table successor is discretized in both the deployment and training
    # forward pass. Straight-through gradients train its probability entries.
    future = straight_through(future)
    values = []
    for action in range(future.shape[1]):
        values.append(read_scalar(program, future[:, action], "values", pairs))
    # Collision likelihood is trained from measured outcomes, not allowed to
    # become an unconstrained action-policy lookup through imitation gradients.
    scores = gamma * torch.stack(values, -1) - step_cost - collision_cost * risk_all(program, state).detach()
    stop = 2 * read_scalar(program, state, "goals", pairs) - 1
    return torch.cat([stop[:, None], scores[:, 1:]], -1)
