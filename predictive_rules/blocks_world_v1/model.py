"""Blocks image-goal adapter for the current full-table symbolic world model.

The table decoder, source-specific visual attention, categorical grounding,
and proof-based action loss are inherited from State512RuleWorld. Blocks uses
one fully observed image rather than a navigation history, and a goal RGB
image rather than a chair/table category. No original Blocks operators load.
"""
import torch
from torch import nn
from predictive_nesy.state512_rule_world import State512RuleWorld, Config


class BlocksRuleWorld(State512RuleWorld):
    def __init__(self):
        super().__init__(Config(states=512, actions=19, tasks=1))
        # Fully observed fixed-camera domain: no accumulated visual memory.
        nn.init.zeros_(self.memory_projection.weight)
        self.memory_projection.requires_grad_(False)

    def prepare_pair(self, codes, visual, goal_codes, goal_visual):
        n = len(codes)
        task = torch.zeros(n, dtype=torch.long, device=codes.device)
        previous = torch.full((n, 1), 19, dtype=torch.long, device=codes.device)
        lengths = torch.ones(n, dtype=torch.long, device=codes.device)
        p = self.prepare(task, codes[:, None], previous, lengths, start=0,
                         boundary_features=visual[:, None])
        # A separate neural goal head must learn the anonymous goal ID. The
        # goal-image grounder is a training target, never supplied to planning.
        value = self.goal_frame(goal_visual.float().flatten(-2))
        value = value + self.action_embed(previous[:, 0])
        with torch.autocast(device_type=codes.device.type, enabled=False):
            hidden, _ = self.goal_history(value.float()[:, None], self.task(task).float()[None])
        goal_context = self.context_norm(self.context_read(self.embed(goal_codes)))
        goal_context = goal_context + self.goal_delta(hidden[:, 0])
        p['subgoal_logits'] = self.subgoal(goal_context).float()[:, None]
        return p

    def generate_pair(self, *args):
        p = self.prepare_pair(*args)
        return {**{k: v for k, v in p.items() if not k.startswith('rule_')},
                'transitions': self.rows(p)}

    def representation_parameters(self):
        prefixes = ('tokens.', 'observation.', 'context_read.', 'history.',
                    'context_norm.', 'definitions.', 'task.', 'action_embed.')
        return [(n, p) for n, p in self.named_parameters() if n.startswith(prefixes)]

    def set_stage(self, stage):
        for p in self.parameters():
            p.requires_grad_(False)
        if stage == 'representation':
            for _, p in self.representation_parameters():
                p.requires_grad_(True)
        elif stage == 'rules':
            prefixes = ('definition_rules.', 'rule_evidence.', 'raw_', 'goal_', 'subgoal.')
            for n, p in self.named_parameters():
                p.requires_grad_(n.startswith(prefixes))
        else:
            raise ValueError(stage)
