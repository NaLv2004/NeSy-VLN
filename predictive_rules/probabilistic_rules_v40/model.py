"""Learned categorical observation / memory, with separately generated rules.

Images never reach the interpreter. Training-only inverse/reconstruction heads
have no runtime call sites. The state vocabulary is anonymous and learned.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    observations: int = 64
    memories: int = 32
    actions: int = 17
    width: int = 128
    context: int = 128


def hard(logits):
    probabilities = logits.float().softmax(-1)
    discrete = F.one_hot(probabilities.argmax(-1), probabilities.shape[-1]).float()
    return discrete + probabilities - probabilities.detach()


class FactoredWorld(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        c = self.cfg
        self.visual = nn.Sequential(nn.LayerNorm(49*128), nn.Linear(49*128,512), nn.SiLU(), nn.Linear(512,c.width))
        self.observation_definitions = nn.Parameter(torch.randn(c.observations,c.width)*.1)
        self.memory_definitions = nn.Parameter(torch.randn(c.memories,c.width)*.1)
        # The recurrent carrier is a categorical ID, never a continuous GRU
        # hidden vector. Each call decodes that ID, updates it, and quantizes.
        self.memory = nn.GRUCell(c.observations+c.actions+1,c.width)
        self.frame = nn.Linear(49*128,c.context)
        self.context_history = nn.GRU(c.context+c.observations+c.actions+1,c.context,batch_first=True)
        self.task = nn.Embedding(2,c.context)
        self.definition_generator = nn.Linear(c.context,c.memories*c.width)
        self.observation_embedding = nn.Linear(c.observations,c.width,bias=False)
        self.action_embedding = nn.Embedding(c.actions,c.width)
        self.rule = nn.Sequential(nn.Linear(c.width*3+c.context,256),nn.SiLU(),nn.Linear(256,c.observations))
        self.goal = nn.Sequential(nn.Linear(c.context,256),nn.SiLU(),nn.Linear(256,c.memories+c.observations))
        self.renewal = nn.Sequential(nn.Linear(c.memories+c.observations*2+1,128),nn.SiLU(),nn.Linear(128,1))
        # Auxiliary heads train abstractions, never select runtime actions.
        # Observation codes ONLY. A future memory code contains action history
        # and would leak the answer into inverse prediction.
        self.inverse = nn.Sequential(nn.Linear(2*c.observations+10,256),nn.SiLU(),nn.Linear(256,c.actions))
        self.reconstruct = nn.Linear(c.observations,49*128)
        self.memory_reconstruct = nn.Linear(c.memories,c.width)
        # Generic learned persistence, identical mechanism for ALL action IDs.
        # No action is assigned a hand-coded identity/STOP/collision effect.
        self.persistence = nn.Sequential(nn.Linear(c.width*3+c.context,64),nn.SiLU(),nn.Linear(64,1))
        self.action_persistence = nn.Embedding(c.actions,1)
        nn.init.zeros_(self.persistence[-1].weight);nn.init.zeros_(self.persistence[-1].bias)
        nn.init.constant_(self.action_persistence.weight,-3.)
        # A learned operator class shared across ALL possible source IDs.
        # Supervision comes from observed action effects, never action names.
        # Class 0 permits change, class 1 preserves the observed code.
        self.operator_class=nn.Embedding(c.actions,2)
        nn.init.zeros_(self.operator_class.weight)
        # Forward values remain identical. Copy evidence is a categorical
        # premise, not a differentiable way to rename the current state to
        # match an unlikely successor. The recurrent categorical carrier also
        # bounds credit assignment; local transition/perception losses remain.
        self.detach_copy_evidence=True
        self.detach_memory_carrier=True

    def export_config(self):
        return asdict(self.cfg)

    def observe(self, features):
        z = F.normalize(self.visual(features.float().flatten(-2)),dim=-1)
        return 12 * z @ F.normalize(self.observation_definitions,dim=-1).T

    def memory_step(self, memory, observation, previous, initial=False):
        """U(m, observed_current, action) with finite-state inputs only.

        The zero initial carrier is a fixed sequence boundary, not a hidden
        recurrent history. All subsequent calls receive hard one-hot memory.
        """
        shape=observation.shape[:-1]
        x=torch.cat([observation,F.one_hot(previous,self.cfg.actions+1).float()],-1)
        carrier=memory.detach() if memory is not None and self.detach_memory_carrier else memory
        hidden=(torch.zeros(*shape,self.cfg.width,device=x.device) if initial
                else self.memory_reconstruct(carrier))
        state=self.memory(x.reshape(-1,x.shape[-1]),hidden.reshape(-1,self.cfg.width)).reshape(*shape,self.cfg.width)
        logits=12*F.normalize(state,dim=-1)@F.normalize(self.memory_definitions,dim=-1).T
        return logits,state

    def memory_sequence(self, observations, previous, hidden=None):
        # `hidden`, if supplied for a counterfactual successor, is a hard memory
        # distribution of width memories, NOT the GRU's continuous activation.
        carrier=hidden[0] if hidden is not None else None
        logits=[];states=[]
        for t in range(observations.shape[1]):
            # M_t summarizes actions and observations available BEFORE O_t.
            # At a new sequence only, O_0 grounds the initial memory. Otherwise
            # M_t=U(M_{t-1}, O_{t-1}, A_{t-1}), exactly as runtime rule execution.
            source_observation=observations[:,max(0,t-1)]
            lm,state=self.memory_step(carrier,source_observation,previous[:,t],initial=carrier is None)
            logits.append(lm);states.append(state);carrier=hard(lm)
        return torch.stack(logits,1),torch.stack(states,1),carrier[None]

    def context_sequence(self, features, observations, previous, task):
        x = torch.cat([self.frame(features.float().flatten(-2)),observations,
                       F.one_hot(previous,self.cfg.actions+1).float()],-1)
        # Observational dynamics cannot change merely because a different task
        # was requested. Task conditioning is reserved for the generated goal.
        return self.context_history(x.float())[0]

    def definitions(self, context):
        return self.definition_generator(context).reshape(-1,self.cfg.memories,self.cfg.width)

    def transition(self, context, memory, observation, action, definitions=None):
        """Selected complete categorical row. Inputs are hard one-hot IDs.

        Differentiable one-hot relaxations are used ONLY for backward gradients.
        Runtime export enumerates every (memory, observation, action) tuple.
        """
        definitions = self.definitions(context) if definitions is None else definitions
        memory_embedding = torch.einsum('b...m,bmw->b...w',memory,definitions)
        observation_embedding = self.observation_embedding(observation)
        action_embedding = self.action_embedding(action)
        c = context.reshape(context.shape[0], *((1,)*(memory.ndim-2)), -1).expand(*memory.shape[:-1],context.shape[-1])
        inputs=torch.cat([memory_embedding,observation_embedding,action_embedding,c],-1)
        lo = self.rule(inputs)
        gates=(self.persistence(inputs)+self.action_persistence(action)).sigmoid()
        evidence=observation.detach() if getattr(self,'detach_copy_evidence',False) else observation
        po=gates*evidence+(1-gates)*lo.softmax(-1)
        # Only the independently measured effect-classification objective
        # trains this operator class. Planning convenience cannot change it.
        preserve=hard(self.operator_class(action)).detach()[...,1,None]
        po=preserve*evidence+(1-preserve)*po
        log_observation=po.clamp_min(1e-9).log()
        # Future memory is determined now, from current hard premises. Future
        # RGB or its eventual observed ID never corrects/replaces this memory.
        # This permits several possible future observed IDs to share one
        # rigorously defined next memory in a stochastic symbolic operator.
        lm,_=self.memory_step(memory,observation,action)
        return lm,log_observation

    def goal_logits(self, context, task):
        return self.goal(context+self.task(task)).split([self.cfg.memories,self.cfg.observations],-1)

    def persistence_logits(self,context,memory,observation,action):
        definitions=self.definitions(context)
        memory_embedding=torch.einsum('b...m,bmw->b...w',memory,definitions)
        c=context.reshape(context.shape[0],*((1,)*(memory.ndim-2)),-1).expand(*memory.shape[:-1],context.shape[-1])
        inputs=torch.cat([memory_embedding,self.observation_embedding(observation),self.action_embedding(action),c],-1)
        return self.persistence(inputs)+self.action_persistence(action)

    def renewal_logits(self, memory, observed, predicted_observation, age):
        return self.renewal(torch.cat([memory,observed,predicted_observation,age[...,None]/32],-1)).squeeze(-1)

    def forward(self, data):
        # A DDP-visible entry point; all training heads participate in one graph.
        from objectives import objective
        return objective(self, data)

    def load_v37(self, state):
        """Reuse weights; change only the recurrent carrier and table factorization."""
        state=dict(state)
        for name in ('weight_ih','weight_hh','bias_ih','bias_hh'):
            state['memory.'+name]=state.pop('memory.'+name+'_l0')
        for name in ('weight','bias'):
            state['rule.2.'+name]=state['rule.2.'+name][self.cfg.memories:]
            state['persistence.2.'+name]=state['persistence.2.'+name][1:2]
        state['action_persistence.weight']=state['action_persistence.weight'][:,1:2]
        self.load_state_dict(state,strict=True)

    @torch.no_grad()
    def compile(self, context, task):
        assert context.shape[0] == 1
        c=self.cfg
        s,o,a=torch.meshgrid(torch.arange(c.memories,device=context.device),
            torch.arange(c.observations,device=context.device),torch.arange(c.actions,device=context.device),indexing='ij')
        s=s.flatten();o=o.flatten();a=a.flatten();definitions=self.definitions(context)
        memories=[];probabilities=[]
        for begin in range(0,len(s),4096):
            sm,ob=self.transition(context,F.one_hot(s[None,begin:begin+4096],c.memories).float(),
                F.one_hot(o[None,begin:begin+4096],c.observations).float(),a[None,begin:begin+4096],definitions)
            memories.append(sm.argmax(-1).squeeze(0).cpu())
            probability=ob.exp()
            # An empirically learned preserving operator has exact singleton
            # support. A log floor is numerical protection, not extra outcomes.
            preserve=self.operator_class(a[None,begin:begin+4096]).argmax(-1)==1
            probability=torch.where(preserve[...,None],F.one_hot(o[None,begin:begin+4096],c.observations).float(),probability)
            probability=probability/probability.sum(-1,keepdim=True)
            probabilities.append(probability.squeeze(0).cpu())
        goal_m,goal_o=self.goal_logits(context,task)
        return dict(next_memory=torch.cat(memories).reshape(c.memories*c.observations,c.actions).numpy(),
                    outcome_probability=torch.cat(probabilities).reshape(c.memories*c.observations,c.actions,c.observations).numpy(),
                    goal=int(goal_m.argmax(-1))*c.observations+int(goal_o.argmax(-1)))
