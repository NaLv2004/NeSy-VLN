"""Train a final visual block through hard tokens over complete trajectories.

Continuous features stay in this perceptual module. Its only public output is
the same hard categorical alphabet accepted by TrainableAlphabetAutomaton.
Canonical microbatches also cover single-frame online observations.
"""
import torch
import torch.nn.functional as F
from torch import nn
from .learned_visual_tokens import LearnedVisualAlphabet, SymbolTokens


class SequencePerception(nn.Module):
    def __init__(self, block, vocabulary, canonical_batch=16):
        super().__init__()
        if canonical_batch < 1:
            raise ValueError('Positive canonical visual microbatch required')
        self.block = block.eval()
        self.canonical_batch = canonical_batch
        for name in ('mean', 'sd', 'axes'):
            self.register_buffer(name, vocabulary[name].detach().clone())
        self.alphabet = LearnedVisualAlphabet(vocabulary['centers']).requires_grad_(False)

    def forward(self, features):
        if features.ndim != 4 or len(features) == 0:
            raise ValueError('Perceptual prefix features must be N x C x H x W')
        ids, values = [], []
        for start in range(0, len(features), self.canonical_batch):
            x = features[start:start + self.canonical_batch]
            count = len(x)
            if count < self.canonical_batch:
                x = torch.cat([x, x.new_zeros((self.canonical_batch-count, *x.shape[1:]))])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
                z = F.adaptive_avg_pool2d(self.block(x), (7, 7)).flatten(2).transpose(1, 2)
            z = F.normalize(((z.float()-self.mean)/self.sd) @ self.axes, dim=-1)
            tokens = self.alphabet(z)
            ids.append(tokens.ids[:count]); values.append(tokens.values[:count])
        return SymbolTokens(torch.cat(ids), torch.cat(values))
