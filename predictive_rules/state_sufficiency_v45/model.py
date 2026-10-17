"""Training-only effect sufficiency; runtime remains the v43 symbolic system."""
from torch import nn
from base_model import FactoredWorld as BaseWorld,Config,hard

class FactoredWorld(BaseWorld):
    def __init__(self,cfg=None):
        super().__init__(cfg);c=self.cfg
        # No continuous frame/context shortcuts: future anonymous symbols must
        # be inferable from the current hard M/O codes and opaque action ID.
        # This is an auxiliary representation objective, never an action head.
        self.effect_aux=nn.Sequential(nn.Linear(c.memories+c.observations+c.actions,256),
            nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,c.memories+c.observations))
