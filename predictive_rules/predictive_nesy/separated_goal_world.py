"""Independent learned rule and goal branches; one immutable symbolic program."""
import copy
import torch
from .raw_boundary_world import RawBoundaryWorld,Config,export_program

class SeparatedGoalWorld(RawBoundaryWorld):
    def __init__(self,config=None):
        super().__init__(config)
        self.goal_frame=copy.deepcopy(self.raw_frame)
        self.goal_history=copy.deepcopy(self.raw_history)
        self.goal_delta=copy.deepcopy(self.raw_delta)

    def generate(self,tasks,history,previous_actions,lengths,*,start,boundary_features,boundary_available=None):
        p=super().generate(tasks,history,previous_actions,lengths,start=start,boundary_features=boundary_features,boundary_available=boundary_available)
        x=self.goal_frame(boundary_features.float().flatten(-2))+self.action_embed(previous_actions)
        with torch.autocast(device_type=x.device.type,enabled=False):
            states,_=self.goal_history(x.float(),self.task(tasks).float()[None])
        raw=states[torch.arange(len(tasks),device=tasks.device),lengths-1];delta=self.goal_delta(raw)
        if boundary_available is not None:delta=delta*boundary_available[:,None].to(delta)
        context=self.context_norm(p['context_raw'])+delta
        p['subgoal_logits']=self.subgoal(context).float()[:,None]
        return p
