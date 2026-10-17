"""Full generated rules and goal with richer causal boundary evidence.

Continuous visual features exist only as neural inputs at rule revisions.
Predicted states, supervision targets and executable rules remain categorical.
No neural perception or visual correction is introduced inside a rule block.
"""
import torch
from torch import nn
from .definition_conditioned_world import DefinitionConditionedWorld,Config,export_program

class RawBoundaryWorld(DefinitionConditionedWorld):
    def __init__(self,config=None):
        super().__init__(config);c=self.cfg
        self.raw_frame=nn.Sequential(nn.LayerNorm(49*128),nn.Linear(49*128,256),nn.GELU(),nn.Linear(256,c.hidden),nn.LayerNorm(c.hidden))
        self.raw_history=nn.GRU(c.hidden,c.hidden,batch_first=True)
        self.raw_delta=nn.Linear(c.hidden,c.hidden,bias=False)
        nn.init.zeros_(self.raw_delta.weight)

    def generate(self,tasks,history,previous_actions,lengths,*,start,boundary_features,boundary_available=None):
        if boundary_features.shape!=(*history.shape[:2],49,128):raise ValueError('Boundary feature shape differs from causal history')
        if not torch.isfinite(boundary_features).all():raise ValueError('Nonfinite visual evidence')
        program=super().generate(tasks,history,previous_actions,lengths,start=start)
        x=self.raw_frame(boundary_features.float().flatten(-2))+self.action_embed(previous_actions)
        with torch.autocast(device_type=x.device.type,enabled=False):
            states,_=self.raw_history(x.float(),self.task(tasks).float()[None])
        raw=states[torch.arange(len(tasks),device=tasks.device),lengths-1]
        delta=self.raw_delta(raw)
        if boundary_available is not None:delta=delta*boundary_available[:,None].to(delta)
        context=self.context_norm(program['context_raw'])+delta
        program['transitions']=self.definition_rules(program['definitions'],context)
        program['subgoal_logits']=self.subgoal(context).float()[:,None]
        return program
