from dataclasses import dataclass, fields
import torch


@dataclass
class Program:
    definitions: torch.Tensor
    premises: torch.Tensor
    state_heads: torch.Tensor
    action_heads: torch.Tensor
    effects: torch.Tensor
    action_conditions: torch.Tensor
    weights: torch.Tensor
    version: int = 0

    def map(self, fn):
        return Program(**{f.name: fn(getattr(self, f.name)) if f.name != 'version'
                          else self.version for f in fields(self)})

    def detach(self):
        return self.map(lambda x: x.detach())

    def cpu(self):
        return self.map(lambda x: x.detach().float().cpu())

    def to(self, device):
        return self.map(lambda x: x.to(device))

    def record(self):
        return {f.name: getattr(self, f.name).detach().float().cpu().tolist()
                if f.name != 'version' else self.version for f in fields(self)}

    def validate(self):
        for f in fields(self):
            if f.name == 'version': continue
            x = getattr(self, f.name)
            if not torch.isfinite(x).all(): raise FloatingPointError(f.name)
        for name in ['premises', 'state_heads', 'action_heads', 'effects', 'action_conditions']:
            x = getattr(self, name)
            if (x < 0).any() or not torch.allclose(x.sum(-1), torch.ones_like(x.sum(-1)), atol=1e-4):
                raise ValueError('Invalid selector: ' + name)

    def harden(self):
        import torch.nn.functional as F
        return Program(self.definitions, *[F.one_hot(getattr(self, name).argmax(-1),
            getattr(self, name).shape[-1]).float() for name in
            ['premises', 'state_heads', 'action_heads', 'effects', 'action_conditions']],
            self.weights, self.version)


def describe(program, limit=48):
    p = program.cpu(); rows = []
    for r in range(min(limit, p.weights.numel())):
        rows.append({'id': r, 'weight': round(float(p.weights[r]), 4),
            'premise_indices': p.premises[r].argmax(-1).tolist(),
            'premise_peaks': p.premises[r].max(-1).values.tolist(),
            'state_head': int(p.state_heads[r].argmax()),
            'action_head': int(p.action_heads[r].argmax()),
            'effect_head': int(p.effects[r].argmax()),
            'action_condition': int(p.action_conditions[r].argmax())})
    return {'version': p.version, 'literal_convention': '0=true; 1..K=positive; K+1..2K=negative',
            'rules': rows}
