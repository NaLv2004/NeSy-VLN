"""Compute the same categorical rows without the unused legacy O decoder."""
import types
import torch
from torch.nn import functional as F


def transition(self, context, memory, observation, action, definitions=None):
    b=len(context);shape=observation.shape
    definitions=self.definitions(context) if definitions is None else definitions
    md=torch.einsum('b...m,bmw->b...w',memory,definitions)
    ae=self.action_embedding(action)
    oe=self.observation_embedding(observation)
    ctx=context.reshape(b,*((1,)*(memory.ndim-2)),-1).expand(*memory.shape[:-1],context.shape[-1])
    inputs=torch.cat([md,oe,ae,ctx],-1)
    carrier=memory.detach() if self.detach_memory_carrier else memory
    hidden=self.memory_reconstruct(carrier)
    x=torch.cat([observation,F.one_hot(action,self.cfg.actions+1).float()],-1)
    state=self.memory_predictor(x.reshape(-1,x.shape[-1]),hidden.reshape(-1,self.cfg.width)).reshape(*memory.shape[:-1],self.cfg.width)
    state=state+self.memory_rule(inputs)
    lm=12*F.normalize(state,dim=-1)@F.normalize(self.memory_definitions,dim=-1).T
    obsdefs=F.normalize(self.observation_definitions[None]+self.observation_definition_delta(context),dim=-1)
    preserve=(self.operator_class(action).argmax(-1)==1)[...,None].float()
    lo=self.definition_rule(md.reshape(b,-1,self.cfg.width),
        observation.reshape(b,-1,self.cfg.observations),obsdefs,
        ae.reshape(b,-1,self.cfg.width),context,preserve.reshape(b,-1,1)).reshape(shape)
    return lm,lo


def install(model):
    model.transition=types.MethodType(transition,model)
    return model
