"""Definition-aware categorical rules; no continuous runtime-state carrier."""
import torch
from base_model import FactoredWorld as BaseWorld, Config, hard
from definition_rule import DefinitionRule, definition_inputs


class FactoredWorld(BaseWorld):
    def __init__(self,cfg=None):
        super().__init__(cfg)
        self.definition_rule=DefinitionRule(self.cfg.width,self.cfg.context)
        # The old index-specific output decoder is retained only for exact
        # checkpoint provenance. It is neither optimized nor used for O'.
        self.rule.requires_grad_(False)

    def load_initial(self,state,head):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected
        expected={k for k in self.state_dict() if k.startswith(('temporal_contrast.','definition_rule.'))}
        assert set(missing)==expected,(missing,expected)
        self.definition_rule.load_state_dict(head,strict=True)

    def transition(self,context,memory,observation,action,definitions=None):
        # M' still comes from current integer premises, independently of
        # future observations. No actual post-action image is accepted here.
        next_memory,_=super().transition(context,memory,observation,action,definitions)
        b=len(context);shape=observation.shape
        arguments=definition_inputs(self,context,memory.reshape(b,-1,self.cfg.memories),
            observation.reshape(b,-1,self.cfg.observations),action.reshape(b,-1))
        logp=self.definition_rule(*arguments).reshape(shape)
        return next_memory,logp
