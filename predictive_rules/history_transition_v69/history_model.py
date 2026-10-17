"""Predict a future anonymous observation from discrete observation/action history.

This network has no action-score output. Candidate action IDs are inputs. All
hidden activations are recomputed from finite symbolic histories, never carried
between physical actions as an unrecorded state.
"""
import torch
from torch import nn
from torch.nn import functional as F

WINDOW = 8

class HistoryTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Embedding(18,32)
        self.cell = nn.GRUCell(160,192)
        self.query = nn.Sequential(nn.LayerNorm(192+128*3),nn.Linear(192+128*3,384),
                                   nn.SiLU(),nn.Linear(384,128))
        nn.init.zeros_(self.query[-1].weight);nn.init.zeros_(self.query[-1].bias)

    def forward(self,observations,previous,mask,definitions,memory_definition,
                candidate_action,context,base_logp,preserve):
        # IDs are interpreted against ONE fixed program's generated definitions.
        evidence=definitions.gather(1,observations[...,None].expand(-1,-1,128))
        x=torch.cat([evidence,self.action(previous)],-1)
        hidden=torch.zeros(len(x),192,device=x.device,dtype=x.dtype)
        for t in range(x.shape[1]):
            updated=self.cell(x[:,t],hidden)
            hidden=torch.where(mask[:,t,None],updated,hidden)
        query=self.query(torch.cat([hidden,memory_definition,candidate_action,context],-1))
        residual=torch.einsum('bw,bow->bo',query,definitions)
        logp=(base_logp+residual).log_softmax(-1)
        copied=F.one_hot(observations[:,-1],64).float()
        p=torch.where(preserve[:,None],copied,logp.exp())
        return p.clamp_min(1e-30).log()

def history_window(observations,previous,times):
    positions=times[:,None]+torch.arange(1-WINDOW,1,device=times.device)[None]
    mask=positions>=0;indices=positions.clamp_min(0)
    return observations.gather(1,indices),previous.gather(1,indices),mask

