"""Action-conditioned prediction of learned discrete visual patch symbols.

There are no named semantic states, rewards, geometry, task-success labels or
policy outputs. This is the dynamics component of a boundary-generated program.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn

@dataclass
class WorldConfig:
    patches:int=49
    vocabulary:int=64
    actions:int=17
    width:int=256
    heads:int=8
    layers:int=4
    feedforward:int=1024

class DiscreteVisualWorld(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.cfg=config or WorldConfig();c=self.cfg
        self.symbol=nn.Embedding(c.vocabulary,c.width)
        self.position=nn.Parameter(torch.zeros(1,c.patches,c.width))
        self.action=nn.Embedding(c.actions,c.width)
        layer=nn.TransformerEncoderLayer(c.width,c.heads,c.feedforward,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,c.layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(c.width);self.output=nn.Linear(c.width,c.vocabulary)
        nn.init.normal_(self.position,std=.02)
    def forward(self,symbols,actions):
        if symbols.dtype!=torch.long or symbols.shape[-1]!=self.cfg.patches:raise ValueError('Wrong discrete visual input')
        if actions.shape!=(len(symbols),):raise ValueError('One action ID per predecessor state')
        x=self.symbol(symbols)+self.position+self.action(actions)[:,None]
        return self.output(self.norm(self.encoder(x)))
    def export_config(self):return asdict(self.cfg)
