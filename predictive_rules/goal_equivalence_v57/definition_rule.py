"""Anonymous-code transition decoder that consumes the actual definitions.

This module generates a categorical row from integer premises. It is not a
planner and never carries a continuous state between simulated actions.
Renaming observation IDs and their definitions must only rename its output.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DefinitionRule(nn.Module):
    def __init__(self, width=128, context=128):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(width * 3 + context, 256),
                                   nn.SiLU(), nn.Linear(256, width))
        self.persistence = nn.Sequential(nn.Linear(width * 3 + context, 64),
                                         nn.SiLU(), nn.Linear(64, 1))
        nn.init.constant_(self.persistence[-1].bias, -3.)
        self.log_scale = nn.Parameter(torch.tensor(2.48490665))  # log(12)

    def forward(self, memory_definition, observation, definitions,
                action_embedding, context, preserve):
        """All premise values are hard IDs; their definitions are fixed rules.

        Shapes: [B,A,W], [B,A,O], [B,O,W], [B,A,W], [B,C], [B,A,1].
        Returning log categorical probabilities permits exact table export.
        """
        evidence = torch.einsum('bao,bow->baw', observation, definitions)
        ctx = context[:, None].expand(-1, observation.shape[1], -1)
        inputs = torch.cat([memory_definition, evidence, action_embedding, ctx], -1)
        query = F.normalize(self.query(inputs), dim=-1)
        scale = self.log_scale.exp().clamp(1., 32.)
        logits = scale * torch.einsum('baw,bow->bao', query, definitions)
        gate = self.persistence(inputs).sigmoid()
        # A copied premise is an integer fact, not a differentiable shortcut
        # that can rename the input so a wrong transition appears correct.
        copied = observation.detach()
        probabilities = gate * copied + (1-gate) * logits.softmax(-1)
        probabilities = preserve * copied + (1-preserve) * probabilities
        return probabilities.clamp_min(1e-9).log()


def definition_inputs(model, context, memory, observation, actions):
    definitions = F.normalize(model.observation_definitions[None] +
                              model.observation_definition_delta(context), dim=-1)
    memory_definition = torch.einsum('bam,bmw->baw', memory, model.definitions(context))
    action_embedding = model.action_embedding(actions)
    preserve = (model.operator_class(actions).argmax(-1) == 1)[..., None].float()
    return memory_definition, observation, definitions, action_embedding, context, preserve
