"""Action-indexed bilinear scores between anonymous learned definitions.

Inspired by RESCAL's directed entity/relation/entity score, not a reproduction
of its tensor-factorization objective. This head emits categorical rule rows.
Continuous definition vectors are fixed program parameters, never a rolling
world state or a neural planning trajectory.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class RelationalRule(nn.Module):
    def __init__(self, base, actions=17, width=128):
        super().__init__()
        self.base=copy.deepcopy(base)
        self.observation_relation=nn.Parameter(torch.zeros(actions,width,width))
        self.memory_relation=nn.Parameter(torch.zeros(actions,width,width))

    def forward(self,memory_definition,observation,definitions,action_embedding,
                context,preserve,action_ids):
        assert action_ids.shape==observation.shape[:2]
        evidence=torch.einsum('bao,bow->baw',observation,definitions)
        ctx=context[:,None].expand(-1,observation.shape[1],-1)
        inputs=torch.cat([memory_definition,evidence,action_embedding,ctx],-1)
        query=F.normalize(self.base.query(inputs),dim=-1)
        # The matrices are fully learned for opaque action IDs. No spatial
        # direction, inverse relation or identity is assigned by hand.
        relation=(torch.einsum('baw,bawv->bav',evidence,self.observation_relation[action_ids])+
            torch.einsum('baw,bawv->bav',F.normalize(memory_definition,dim=-1),self.memory_relation[action_ids]))
        logits=self.base.log_scale.exp().clamp(1.,32.)*torch.einsum('baw,bow->bao',query+relation,definitions)
        gate=self.base.persistence(inputs).sigmoid();copied=observation.detach()
        probabilities=gate*copied+(1-gate)*logits.softmax(-1)
        probabilities=preserve*copied+(1-preserve)*probabilities
        return probabilities.clamp_min(1e-9).log()
