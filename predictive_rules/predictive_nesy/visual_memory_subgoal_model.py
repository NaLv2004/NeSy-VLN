"""Anonymous history-grounded states, without future action tags or clock states.

Neural history grounding runs at rule revisions and for training labels only.
Repeated visual codes leave memory and its grounded state exactly unchanged.
The executable program remains only definitions, transitions and a discrete goal.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .visual_graph_subgoal_model import VisualGraphSubgoalModel, Config


class VisualMemorySubgoalModel(VisualGraphSubgoalModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.visual_memory = nn.GRUCell(self.cfg.width, self.cfg.width)
        self.memory_projection = nn.Linear(self.cfg.width, self.cfg.width, bias=False)
        nn.init.zeros_(self.memory_projection.weight)

    def advance(self, memory, previous, previous_codes, feature, codes):
        changed = codes.ne(previous_codes).any(-1, keepdim=True)
        # Exact repeated-code equality avoids different GEMM batch shapes
        # creating an artificial memory update from floating-point roundoff.
        feature = torch.where(changed, feature, previous)
        delta = feature - previous
        gate = -torch.expm1(-delta.float().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt())
        proposal = self.visual_memory(delta, memory)
        memory = torch.where(changed, memory + gate.to(memory)*(proposal-memory), memory)
        return memory, feature, codes

    def ground_features(self, program, codes, memory=None):
        features = self.observation(self.embed(codes))
        if memory is None:
            memory = torch.zeros_like(features)
        features = F.normalize(features+self.memory_projection(memory), dim=-1)
        definitions = F.normalize(program['definitions'], dim=-1)
        return (8*torch.einsum('bd,bkd->bk', features, definitions).float()).softmax(-1)[:, None]

    def generate(self, tasks, history, previous_actions, lengths, *, start):
        program = super().generate(tasks, history, previous_actions, lengths, start=start)
        features = self.observation(self.embed(history))
        memory = torch.zeros_like(features[:, 0]); previous = features[:, 0]; old_codes = history[:, 0]
        for t in range(1, history.shape[1]):
            updated, new_feature, new_codes = self.advance(memory, previous, old_codes, features[:, t], history[:, t])
            valid = (t < lengths)[:, None]
            memory = torch.where(valid, updated, memory)
            previous = torch.where(valid, new_feature, previous)
            old_codes = torch.where(valid, new_codes, old_codes)
        program.update(ground_memory=memory, ground_last=previous, ground_codes=old_codes)
        return program

    def ground(self, program, codes):
        if not torch.equal(codes, program['ground_codes']):
            raise ValueError('Runtime grounding must use the current revision image')
        return self.ground_features(program, codes, program['ground_memory'])

    def encode_sequence(self, program, codes):
        features = self.observation(self.embed(codes))
        memory = program['ground_memory']; previous = program['ground_last']; old_codes = program['ground_codes']
        states = []
        for t in range(codes.shape[1]):
            memory, previous, old_codes = self.advance(memory, previous, old_codes, features[:, t], codes[:, t])
            projected = F.normalize(previous+self.memory_projection(memory), dim=-1)
            logits = torch.einsum('bd,bkd->bk', projected, F.normalize(program['definitions'], dim=-1))
            states.append((8*logits.float()).softmax(-1))
        return torch.stack(states, 1), memory, previous, old_codes

    def training_targets(self, program, codes, actions):
        # Actions never enter future-state labels. They enter transition
        # supervision separately, through the rule premise's action symbol.
        return self.encode_sequence(program, codes)[0][:, :, None]


def export_program(generated, index=0):
    from .graph_subgoal_interpreter import export_program as export_arrays
    keys={'definitions','transitions','subgoal_logits','start','context_raw'}
    if set(generated) != keys|{'ground_memory','ground_last','ground_codes'}:
        raise ValueError('Unexpected visual-memory program fields')
    # Neural grounding intermediates are deliberately absent from execution.
    return export_arrays({key:generated[key] for key in keys}, index)
