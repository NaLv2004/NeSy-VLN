"""Complete neural transition tables conditioned on learned state definitions.

No row ID is assigned a semantic meaning. The source and destination definition
vectors participate in every generated rule. The existing learned goal head
and grounded state representation are preserved when initialized from v11.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .visual_memory_subgoal_model import VisualMemorySubgoalModel,Config,export_program

class DefinitionConditionedTransitions(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.action=nn.Embedding(c.actions,c.width)
        self.context=nn.Linear(c.hidden,c.width)
        self.premise=nn.Linear(c.width,c.width)
        self.query=nn.Sequential(nn.Linear(c.width,c.decoder_hidden),nn.GELU(),nn.Linear(c.decoder_hidden,c.width))
        self.key=nn.Sequential(nn.Linear(c.width,c.width),nn.GELU(),nn.LayerNorm(c.width))
        self.log_scale=nn.Parameter(torch.tensor(2.0794415))
    def forward(self,definitions,context):
        keys=F.normalize(self.key(definitions),dim=-1)
        x=self.premise(definitions)[:,:,None]+self.action.weight[None,None]+self.context(context)[:,None,None]
        query=F.normalize(self.query(F.gelu(x)),dim=-1)
        logits=self.log_scale.exp().clamp(1,64)*torch.einsum('bsad,bkd->bsak',query,keys).float()
        return logits.softmax(-1)

class DefinitionConditionedWorld(VisualMemorySubgoalModel):
    def __init__(self,config=None):
        super().__init__(config)
        del self.transitions
        self.definition_rules=DefinitionConditionedTransitions(self.cfg)
    def generate(self,tasks,history,previous_actions,lengths,*,start):
        if start<0 or start%10 or history.shape[1]>min(start+1,11):raise ValueError('Noncausal program generation')
        if previous_actions.shape!=history.shape[:2]:raise ValueError('Action history shape mismatch')
        x=self.context_read(self.embed(history))+self.action_embed(previous_actions)
        with torch.autocast(device_type=x.device.type,enabled=False):
            states,_=self.history(x.float(),self.task(tasks).float()[None])
        raw=states[torch.arange(len(tasks),device=tasks.device),lengths-1];context=self.context_norm(raw)
        definitions=self.definitions(context);rules=self.definition_rules(definitions,context)
        features=self.observation(self.embed(history));memory=torch.zeros_like(features[:,0])
        previous=features[:,0];old_codes=history[:,0]
        for t in range(1,history.shape[1]):
            updated,new_feature,new_codes=self.advance(memory,previous,old_codes,features[:,t],history[:,t])
            valid=(t<lengths)[:,None];memory=torch.where(valid,updated,memory)
            previous=torch.where(valid,new_feature,previous);old_codes=torch.where(valid,new_codes,old_codes)
        return dict(definitions=definitions,transitions=rules,subgoal_logits=self.subgoal(context).float()[:,None],
            context_raw=raw,start=start,ground_memory=memory,ground_last=previous,ground_codes=old_codes)
