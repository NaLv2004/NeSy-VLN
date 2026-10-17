"""Full transition rules whose inputs include the generated state definitions.

Renaming anonymous state IDs renames rows, columns and the goal together.
No symbol has an assigned physical meaning and no action/value head exists.
All neural computation occurs when the complete rule program is generated.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .visual_memory_subgoal_model import VisualMemorySubgoalModel, Config, export_program


class DefinitionBoundRules(nn.Module):
    def __init__(self, config):
        super().__init__();c=config
        self.action=nn.Embedding(c.actions,c.width)
        self.context=nn.Linear(c.hidden,c.width)
        self.premise=nn.Linear(c.width,c.width)
        self.query=nn.Sequential(nn.Linear(c.width,c.decoder_hidden),nn.GELU(),nn.Linear(c.decoder_hidden,c.width))
        self.key=nn.Sequential(nn.Linear(c.width,c.width),nn.GELU(),nn.LayerNorm(c.width))
        self.goal=nn.Sequential(nn.Linear(c.hidden,c.decoder_hidden),nn.GELU(),nn.Linear(c.decoder_hidden,c.width))
        self.log_scale=nn.Parameter(torch.tensor(2.0794415))

    def forward(self, definitions, context):
        keys=F.normalize(self.key(definitions),dim=-1)
        premises=self.premise(definitions)[:,:,None]+self.action.weight[None,None]+self.context(context)[:,None,None]
        queries=F.normalize(self.query(F.gelu(premises)),dim=-1)
        scale=self.log_scale.exp().clamp(1,64)
        transitions=(scale*torch.einsum('bsad,bkd->bsak',queries,keys).float()).softmax(-1)
        goal_query=F.normalize(self.goal(context),dim=-1)
        goal_logits=scale*torch.einsum('bd,bkd->bk',goal_query,keys).float()
        return transitions,goal_logits[:,None]


class DefinitionBoundModel(VisualMemorySubgoalModel):
    def __init__(self,config=None):
        super().__init__(config)
        del self.transitions,self.subgoal
        self.definition_rules=DefinitionBoundRules(self.cfg)

    def generate(self,tasks,history,previous_actions,lengths,*,start):
        if start<0 or start%10 or history.shape[1]>min(start+1,11):
            raise ValueError('Program generation outside a causal revision boundary')
        if previous_actions.shape!=history.shape[:2]:
            raise ValueError('Previous actions do not match observed history')
        x=self.context_read(self.embed(history))+self.action_embed(previous_actions)
        states,_=self.history(x,self.task(tasks)[None])
        raw=states[torch.arange(len(tasks),device=tasks.device),lengths-1]
        context=self.context_norm(raw);definitions=self.definitions(context)
        transitions,subgoal_logits=self.definition_rules(definitions,context)
        features=self.observation(self.embed(history))
        memory=torch.zeros_like(features[:,0]);previous=features[:,0];old_codes=history[:,0]
        for t in range(1,history.shape[1]):
            updated,new_feature,new_codes=self.advance(memory,previous,old_codes,features[:,t],history[:,t])
            valid=(t<lengths)[:,None]
            memory=torch.where(valid,updated,memory)
            previous=torch.where(valid,new_feature,previous)
            old_codes=torch.where(valid,new_codes,old_codes)
        return dict(definitions=definitions,transitions=transitions,subgoal_logits=subgoal_logits,
            context_raw=raw,start=start,ground_memory=memory,ground_last=previous,ground_codes=old_codes)
