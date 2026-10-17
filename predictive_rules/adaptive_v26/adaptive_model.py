"""Hard observation-conditioned rules with a learned renewal GRU.

The perception boundary exports one anonymous categorical ID. The executable
correction C[predicted, observed] and action T[state, action] are both argmax
tables generated at a renewal. Only the renewal policy runs between programs;
it consumes integer symbols and never chooses a navigation action.
"""
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.state512_rule_world import State512RuleWorld, Config


class ObservationCorrection(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.predicted = nn.Linear(width, width, bias=False)
        self.observed = nn.Linear(width, width, bias=False)
        self.destination = nn.Linear(width, width, bias=False)
        self.context = nn.Linear(hidden, width, bias=False)
        nn.init.zeros_(self.predicted.weight)
        nn.init.zeros_(self.context.weight)
        nn.init.eye_(self.observed.weight)
        nn.init.eye_(self.destination.weight)
        self.log_scale = nn.Parameter(torch.tensor(2.0794415))

    def parameters_for_program(self, definitions, context):
        return (self.predicted(definitions), self.observed(definitions),
                F.normalize(self.destination(definitions), dim=-1), self.context(context),
                self.log_scale.exp().clamp(1, 64))

    def selected(self, definitions, context, predicted, observed):
        before, evidence, keys, bias, scale = self.parameters_for_program(definitions, context)
        index = torch.arange(len(predicted), device=predicted.device)
        query = F.normalize(before[index, predicted] + evidence[index, observed] + bias, dim=-1)
        return scale * torch.einsum('bd,bkd->bk', query, keys).float()

    @torch.no_grad()
    def compile(self, definitions, context, chunk=8):
        before, evidence, keys, bias, scale = self.parameters_for_program(definitions, context)
        result = []
        for start in range(0, evidence.shape[1], chunk):
            query = F.normalize(before[:, :, None] + evidence[:, None, start:start+chunk] + bias[:, None, None], dim=-1)
            logits = scale * torch.einsum('bsod,bkd->bsok', query, keys).float()
            result.append(logits.argmax(-1))
        return torch.cat(result, -1)


class RevisionGRU(nn.Module):
    def __init__(self, states=512, actions=17, tasks=2, hidden=128):
        super().__init__()
        self.symbol = nn.Embedding(states, 64)
        self.action = nn.Embedding(actions+1, 32)
        self.task = nn.Embedding(tasks, 32)
        self.recurrent = nn.GRUCell(257, hidden)
        self.decision = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.constant_(self.decision[-1].bias, -2.)
        self.hidden = hidden

    def step(self, observed, predicted, previous_action, task, age, hidden, goal=None):
        for ids in (observed, predicted):
            if ids.dtype != torch.long or ids.ndim != 1 or ((ids<0)|(ids>=512)).any():
                raise TypeError('Revision GRU accepts scalar categorical observation IDs only')
        if goal is None:goal=predicted
        values = torch.cat([self.symbol(observed), self.symbol(predicted), self.symbol(goal), self.action(previous_action),
                            self.task(task), age.float()[:, None]/32.], -1)
        hidden = self.recurrent(values.float(), hidden.float())
        return self.decision(hidden).squeeze(-1), hidden

    def sequence(self, observed, predicted, previous_actions, tasks, ages, lengths, goal=None):
        hidden = torch.zeros(len(tasks), self.hidden, device=tasks.device)
        logits = torch.zeros(len(tasks), device=tasks.device)
        for t in range(observed.shape[1]):
            value, following = self.step(observed[:, t], predicted[:, t], previous_actions[:, t], tasks, ages[:, t], hidden, goal=goal)
            valid = t < lengths
            hidden = torch.where(valid[:, None], following, hidden)
            logits = torch.where(valid, value, logits)
        return logits


class AdaptiveObservedWorld(State512RuleWorld):
    def __init__(self, config=None):
        super().__init__(config or Config(states=512))
        self.correction = ObservationCorrection(self.cfg.width, self.cfg.hidden)
        self.revision = RevisionGRU(states=self.cfg.states, actions=self.cfg.actions, tasks=2)

    def prepare(self, tasks, history, previous_actions, lengths, *, start, boundary_features, boundary_available=None):
        if not 0 <= start < 32 or history.shape[1] > min(start+1, 11):
            raise ValueError('Noncausal adaptive program inputs')
        # Base numerical computation is independent of its ten-step clock;
        # remove only that historical validation restriction.
        proxy = 0 if start == 0 else 10
        prepared = super().prepare(tasks, history, previous_actions, lengths, start=proxy,
                                   boundary_features=boundary_features, boundary_available=boundary_available)
        prepared['start'] = start
        return prepared

    def generate(self, *args, **kwargs):
        p = self.prepare(*args, **kwargs)
        table = self.rows(p)
        return {**{k: v for k, v in p.items() if not k.startswith('rule_')},
                'transitions': table, 'correction_table': self.correction.compile(p['definitions'], p['context_raw'])}
