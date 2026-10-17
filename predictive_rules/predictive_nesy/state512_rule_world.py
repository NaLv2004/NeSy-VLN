"""Full generated 512-state rules, with equivalent selected-row training.

At inference the neural generator exports every rule. During supervised
learning only observed premise rows require gradients; a detached complete
table supplies the unchanged exact-goal proof loss. No sparse or partial rule
program is sent to the executable controller.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .evidence_rule_world import EvidenceRuleWorld,EvidenceTransitions,Config,export_program


class RowEvidenceTransitions(EvidenceTransitions):
    def rows(self,definitions,context,evidence,source_ids=None):
        source=definitions if source_ids is None else definitions.gather(1,source_ids[:,:,None].expand(-1,-1,definitions.shape[-1]))
        keys=F.normalize(self.key(definitions),dim=-1)
        x=self.premise(source)[:,:,None]+self.action.weight[None,None]+self.context(context)[:,None,None]
        query=F.normalize(self.query(F.gelu(x+evidence[:,:,None])),dim=-1)
        logits=self.log_scale.exp().clamp(1,64)*torch.einsum('bsad,bkd->bsak',query,keys).float()
        return logits.softmax(-1)


class State512RuleWorld(EvidenceRuleWorld):
    def __init__(self,config=None):
        super().__init__(config or Config(states=512))
        self.definition_rules=RowEvidenceTransitions(self.cfg)

    def prepare(self,tasks,history,previous_actions,lengths,*,start,boundary_features,boundary_available=None):
        if start<0 or start%10 or history.shape[1]>min(start+1,11):raise ValueError('Noncausal program generation')
        if previous_actions.shape!=history.shape[:2]:raise ValueError('Action history shape mismatch')
        if boundary_features.shape!=(*history.shape[:2],49,128):raise ValueError('Boundary feature shape mismatch')
        if not torch.isfinite(boundary_features).all():raise ValueError('Nonfinite visual evidence')
        indices=torch.arange(len(tasks),device=tasks.device)
        # The frozen categorical grounder must use the same FP32 arithmetic
        # under mixed-precision training and ordinary runtime inference.
        with torch.autocast(device_type=history.device.type,enabled=False):
            x=self.context_read(self.embed(history))+self.action_embed(previous_actions)
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
        return dict(definitions=definitions,subgoal_logits=self.subgoal(goal_context).float()[:,None],context_raw=raw,start=start,
            ground_memory=memory,ground_last=previous,ground_codes=old_codes,rule_context=world_context,
            rule_visual=boundary_features[indices,lengths-1],rule_available=boundary_available)

    def rows(self,prepared,source_ids=None):
        definitions=prepared['definitions']
        source=definitions if source_ids is None else definitions.gather(1,source_ids[:,:,None].expand(-1,-1,definitions.shape[-1]))
        evidence=self.rule_evidence(source,prepared['rule_context'],prepared['rule_visual'])
        if prepared['rule_available'] is not None:evidence=evidence*prepared['rule_available'][:,None,None].to(evidence)
        return self.definition_rules.rows(definitions,prepared['rule_context'],evidence,source_ids)

    def generate(self,*args,**kwargs):
        p=self.prepare(*args,**kwargs);table=self.rows(p)
        return {**{k:v for k,v in p.items() if not k.startswith('rule_')},'transitions':table}

    def encode_sequence(self,program,codes):
        with torch.autocast(device_type=codes.device.type,enabled=False):return super().encode_sequence(program,codes)

    def ground(self,program,codes):
        with torch.autocast(device_type=codes.device.type,enabled=False):return super().ground(program,codes)
