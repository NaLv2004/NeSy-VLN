"""Spatially shared abstraction of an existing discrete visual alphabet.

All inputs are integer cluster IDs. Convolutions reason over those symbols,
not RGB, depth or continuous encoder activations. Predicate meanings and all
transition/action rule tables remain learned, with ten-action program revision.
"""
import math
import torch
from torch import nn
from .probabilistic_automaton import ProbabilisticTokenAutomaton


class SpatialSymbolReader(nn.Module):
    def __init__(self, patches, factors, width):
        super().__init__()
        self.side=math.isqrt(patches)
        if self.side*self.side!=patches:
            raise ValueError('Spatial token positions must form a square grid')
        self.conv=nn.Sequential(nn.Conv2d(8,32,3,padding=1),nn.GELU(),
                                nn.Conv2d(32,64,3,padding=1),nn.GELU(),
                                nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),
                                nn.Linear(64*9,factors*width),nn.LayerNorm(factors*width))

    def forward(self, flat):
        grid=flat.reshape(-1,self.side,self.side,8).permute(0,3,1,2)
        return self.conv(grid)


class SpatialTokenAutomaton(ProbabilisticTokenAutomaton):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.tokens=nn.Embedding(self.vocabulary,8)
        self.observation=SpatialSymbolReader(self.patches,self.factors,self.width)

    def embed(self,codes):
        self.validate_codes(codes)
        return self.tokens(codes).flatten(-2)

    def initialize_common_program(self,source):
        """Copy matching rule/generator parameters; observation fitting is fresh."""
        target=self.state_dict();copied=[]
        for name,value in source.items():
            if name.startswith(('tokens.','observation.')):continue
            if name not in target or target[name].shape!=value.shape:
                raise ValueError(f'Incompatible common program parameter: {name}')
            target[name]=value;copied.append(name)
        self.load_state_dict(target)
        return copied
