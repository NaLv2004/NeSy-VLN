"""Generate complete rule rows using source-conditioned boundary image evidence.

The attention is part of the neural program generator, invoked only at a rule
revision. Its tensors are never exported to the symbolic executor. States and
goal remain the same anonymous categorical variables as the preserved model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .separated_goal_world import SeparatedGoalWorld,Config,export_program
from .definition_conditioned_world import DefinitionConditionedTransitions


class SourceEvidence(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.visual=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,c.width))
        self.position=nn.Embedding(c.patches,c.width)
        self.query=nn.Sequential(nn.LayerNorm(c.width),nn.Linear(c.width,c.width))
        self.context=nn.Linear(c.hidden,c.width)
        self.attention=nn.MultiheadAttention(c.width,4,batch_first=True,dropout=0.)
        self.output=nn.Sequential(nn.LayerNorm(c.width),nn.Linear(c.width,c.width,bias=False))
        nn.init.zeros_(self.output[-1].weight)

    def forward(self,definitions,context,current_visual):
        values=self.visual(current_visual.float())+self.position.weight[None]
        queries=self.query(definitions)+self.context(context)[:,None]
        attended,_=self.attention(queries,values,values,need_weights=False)
        return self.output(attended)


class EvidenceTransitions(DefinitionConditionedTransitions):
    def forward(self,definitions,context,evidence):
        keys=F.normalize(self.key(definitions),dim=-1)
        x=self.premise(definitions)[:,:,None]+self.action.weight[None,None]+self.context(context)[:,None,None]
        x=x+evidence[:,:,None]
        query=F.normalize(self.query(F.gelu(x)),dim=-1)
        logits=self.log_scale.exp().clamp(1,64)*torch.einsum('bsad,bkd->bsak',query,keys).float()
        return logits.softmax(-1)


class EvidenceRuleWorld(SeparatedGoalWorld):
    def __init__(self,config=None):
        super().__init__(config)
        self.definition_rules=EvidenceTransitions(self.cfg)
        self.rule_evidence=SourceEvidence(self.cfg)

    def generate(self,tasks,history,previous_actions,lengths,*,start,boundary_features,boundary_available=None):
        if start<0 or start%10 or history.shape[1]>min(start+1,11):raise ValueError('Noncausal program generation')
        if previous_actions.shape!=history.shape[:2]:raise ValueError('Action history shape mismatch')
        if boundary_features.shape!=(*history.shape[:2],49,128):raise ValueError('Boundary feature shape mismatch')
        if not torch.isfinite(boundary_features).all():raise ValueError('Nonfinite visual evidence')
        indices=torch.arange(len(tasks),device=tasks.device)
        x=self.context_read(self.embed(history))+self.action_embed(previous_actions)
        with torch.autocast(device_type=x.device.type,enabled=False):
            states,_=self.history(x.float(),self.task(tasks).float()[None])
        raw=states[indices,lengths-1];base=self.context_norm(raw)
        definitions=self.definitions(base)
        features=self.observation(self.embed(history));memory=torch.zeros_like(features[:,0])
        previous=features[:,0];old_codes=history[:,0]
        for t in range(1,history.shape[1]):
            updated,new_feature,new_codes=self.advance(memory,previous,old_codes,features[:,t],history[:,t])
            valid=(t<lengths)[:,None];memory=torch.where(valid,updated,memory)
            previous=torch.where(valid,new_feature,previous);old_codes=torch.where(valid,new_codes,old_codes)
        def branch(frame,gru,project):
            values=frame(boundary_features.float().flatten(-2))+self.action_embed(previous_actions)
            with torch.autocast(device_type=values.device.type,enabled=False):
                hidden,_=gru(values.float(),self.task(tasks).float()[None])
            delta=project(hidden[indices,lengths-1])
            if boundary_available is not None:delta=delta*boundary_available[:,None].to(delta)
            return base+delta
        world_context=branch(self.raw_frame,self.raw_history,self.raw_delta)
        goal_context=branch(self.goal_frame,self.goal_history,self.goal_delta)
        evidence=self.rule_evidence(definitions,world_context,boundary_features[indices,lengths-1])
        if boundary_available is not None:evidence=evidence*boundary_available[:,None,None].to(evidence)
        return dict(definitions=definitions,transitions=self.definition_rules(definitions,world_context,evidence),
            subgoal_logits=self.subgoal(goal_context).float()[:,None],context_raw=raw,start=start,
            ground_memory=memory,ground_last=previous,ground_codes=old_codes)
