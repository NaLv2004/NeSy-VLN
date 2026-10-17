"""Compact causal two-symbol state predictor; outputs next O, never actions."""
import torch
from torch import nn
from torch.nn import functional as F
WINDOW=2
class HistoryTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.action=nn.Embedding(18,32)
        self.query=nn.Sequential(nn.LayerNorm(576),nn.Linear(576,192),nn.SiLU(),nn.Dropout(.1),nn.Linear(192,128))
        nn.init.zeros_(self.query[-1].weight);nn.init.zeros_(self.query[-1].bias)
    def forward(self,observations,previous,mask,definitions,memory_definition,candidate_action,context,base_logp,preserve):
        evidence=definitions.gather(1,observations[...,None].expand(-1,-1,128))
        x=torch.cat([evidence,self.action(previous)],-1)*mask[...,None]
        if x.shape[1]<WINDOW:x=F.pad(x,(0,0,WINDOW-x.shape[1],0))
        query=self.query(torch.cat([x[:,-WINDOW:].flatten(1),memory_definition,candidate_action],-1))
        # Limit correction size and preserve the source distribution initially.
        residual=2*torch.tanh(torch.einsum('bw,bow->bo',query,definitions)/2)
        lp=(base_logp+residual).log_softmax(-1)
        copied=F.one_hot(observations[:,-1],64).float()
        return torch.where(preserve[:,None],copied,lp.exp()).clamp_min(1e-30).log()
def history_window(observations,previous,times):
    positions=times[:,None]+torch.arange(1-WINDOW,1,device=times.device)[None]
    mask=positions>=0;indices=positions.clamp_min(0)
    return observations.gather(1,indices),previous.gather(1,indices),mask
