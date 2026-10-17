"""Learn boundary context for dynamics without moving the grounded vocabulary."""
import copy
import torch
from torch import nn
from .definition_conditioned_world import DefinitionConditionedWorld,Config,export_program


class BoundaryRuleContext(nn.Module):
    def __init__(self,model):
        super().__init__()
        for key in ['tokens','context_read','action_embed','task','history','context_norm']:
            setattr(self,key,copy.deepcopy(getattr(model,key)))
        self.register_buffer('offsets',model.offsets.detach().clone())

    def forward(self,tasks,codes,previous_actions,lengths):
        features=self.tokens(codes+self.offsets).flatten(-2)
        x=self.context_read(features)+self.action_embed(previous_actions)
        with torch.autocast(device_type=x.device.type,enabled=False):
            states,_=self.history(x.float(),self.task(tasks).float()[None])
        return self.context_norm(states[torch.arange(len(tasks),device=tasks.device),lengths-1])


class LearnedRuleContextWorld(DefinitionConditionedWorld):
    def __init__(self,config=None):
        super().__init__(config)
        self.rule_context=BoundaryRuleContext(self)

    def reset_rule_context(self):
        self.rule_context=BoundaryRuleContext(self)

    def generate(self,tasks,history,previous_actions,lengths,*,start):
        if start<0 or start%10 or history.shape[1]>min(start+1,11):
            raise ValueError('Noncausal program generation')
        if previous_actions.shape!=history.shape[:2]:
            raise ValueError('Action history shape mismatch')
        x=self.context_read(self.embed(history))+self.action_embed(previous_actions)
        with torch.autocast(device_type=x.device.type,enabled=False):
            states,_=self.history(x.float(),self.task(tasks).float()[None])
        raw=states[torch.arange(len(tasks),device=tasks.device),lengths-1]
        context=self.context_norm(raw)
        definitions=self.definitions(context)
        rule_context=self.rule_context(tasks,history,previous_actions,lengths)
        rules=self.definition_rules(definitions,rule_context)
        features=self.observation(self.embed(history));memory=torch.zeros_like(features[:,0])
        previous=features[:,0];old_codes=history[:,0]
        for t in range(1,history.shape[1]):
            updated,new_feature,new_codes=self.advance(memory,previous,old_codes,features[:,t],history[:,t])
            valid=(t<lengths)[:,None];memory=torch.where(valid,updated,memory)
            previous=torch.where(valid,new_feature,previous);old_codes=torch.where(valid,new_codes,old_codes)
        return dict(definitions=definitions,transitions=rules,subgoal_logits=self.subgoal(context).float()[:,None],
            context_raw=raw,start=start,ground_memory=memory,ground_last=previous,ground_codes=old_codes)
