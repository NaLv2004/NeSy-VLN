"""Train a perceptual alphabet without allowing visual features into reasoning.

SymbolTokens carries exact categorical values in the forward computation.
The continuous relaxation exists only in its training derivative. At deployment
the automaton accepts the ordinary integer IDs and produces identical results.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .factorized_automaton import categorical
from .probabilistic_automaton import ProbabilisticTokenAutomaton


@dataclass(frozen=True)
class SymbolTokens:
    ids: torch.Tensor
    values: torch.Tensor

    @property
    def ndim(self):return self.ids.ndim
    @property
    def shape(self):return self.ids.shape
    @property
    def device(self):return self.ids.device
    def __len__(self):return len(self.ids)
    def __getitem__(self,key):return SymbolTokens(self.ids[key],self.values[key])


class LearnedVisualAlphabet(nn.Module):
    def __init__(self,centers,temperature=.2):
        super().__init__()
        self.centers=nn.Parameter(centers.detach().clone())
        self.register_buffer('original_centers',centers.detach().clone())
        self.projection_delta=nn.Parameter(torch.zeros(centers.shape[1],centers.shape[1],device=centers.device))
        self.temperature=temperature

    def forward(self,vectors):
        if vectors.ndim not in (3,4) or vectors.shape[-1]!=self.centers.shape[-1]:
            raise ValueError('Expected patch features at the perceptual quantizer only')
        projected=vectors+F.linear(vectors,self.projection_delta)
        distance=torch.cdist(projected.flatten(0,-2),self.centers)
        probability=(-distance.square()/self.temperature).softmax(-1).reshape(*vectors.shape[:-1],len(self.centers))
        ids,values=categorical(probability)
        return SymbolTokens(ids,values)

    def drift_penalty(self):
        return self.projection_delta.square().mean()+(self.centers-self.original_centers).square().mean()


class TrainableAlphabetAutomaton(ProbabilisticTokenAutomaton):
    def validate_codes(self,codes):
        if isinstance(codes,SymbolTokens):
            super().validate_codes(codes.ids)
            if codes.values.shape!=(*codes.ids.shape,self.vocabulary):
                raise ValueError('Incorrect categorical token shape')
            hard=F.one_hot(codes.ids,self.vocabulary).to(codes.values)
            if not torch.equal(codes.values.detach(),hard):
                raise ValueError('A continuous visual feature cannot enter symbolic reasoning')
        else:super().validate_codes(codes)

    def embed(self,codes):
        self.validate_codes(codes)
        if not isinstance(codes,SymbolTokens):return super().embed(codes)
        hard=self.tokens(codes.ids+self.offsets).flatten(-2)
        # Forward values are exactly integer-indexed embeddings. Surrogate
        # derivatives train the quantizer, with no duplicate embedding gradient.
        table=self.tokens.weight.detach().reshape(self.patches,self.vocabulary,8)
        relaxed=torch.einsum('...pv,pvd->...pd',codes.values,table).flatten(-2)
        return hard+(relaxed-relaxed.detach())
