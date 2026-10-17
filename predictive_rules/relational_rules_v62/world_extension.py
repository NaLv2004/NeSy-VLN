ReferenceWorld=FactoredWorld

class FactoredWorld(ReferenceWorld):
    def __init__(self,cfg=None):
        super().__init__(cfg)
        self.relation_head=RelationalRule(self.definition_rule,self.cfg.actions,self.cfg.width)
        self.definition_rule.requires_grad_(False)

    def load_candidate(self,base_state,head_state):
        missing,unexpected=self.load_state_dict(base_state,strict=False)
        assert not unexpected and set(missing)=={k for k in self.state_dict() if k.startswith('relation_head.')}
        self.relation_head.load_state_dict(head_state,strict=True)
        for name,value in base_state.items():
            assert torch.equal(value.cpu(),self.state_dict()[name].cpu()),name

    def transition(self,context,memory,observation,action,definitions=None):
        # Keep original independent categorical M dynamics byte-identical.
        next_memory,_=BaseWorld.transition(self,context,memory,observation,action,definitions)
        b=len(context);shape=observation.shape
        arguments=definition_inputs(self,context,memory.reshape(b,-1,self.cfg.memories),
            observation.reshape(b,-1,self.cfg.observations),action.reshape(b,-1))
        probability=self.relation_head(*arguments,action_ids=action.reshape(b,-1)).reshape(shape)
        return next_memory,probability
