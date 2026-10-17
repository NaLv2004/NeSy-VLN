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
        self.memory = nn.GRU(c.observations+c.actions+1,c.width,batch_first=True)
        self.frame = nn.Linear(49*128,c.context)
        self.context_history = nn.GRU(c.context+c.observations+c.actions+1,c.context,batch_first=True)
        self.task = nn.Embedding(2,c.context)
        self.definition_generator = nn.Linear(c.context,c.memories*c.width)
        self.observation_embedding = nn.Linear(c.observations,c.width,bias=False)
        self.action_embedding = nn.Embedding(c.actions,c.width)
        self.rule = nn.Sequential(nn.Linear(c.width*3+c.context,256),nn.SiLU(),nn.Linear(256,c.memories+c.observations))
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
        self.persistence = nn.Sequential(nn.Linear(c.width*3+c.context,64),nn.SiLU(),nn.Linear(64,2))
        self.action_persistence = nn.Embedding(c.actions,2)
        nn.init.zeros_(self.persistence[-1].weight);nn.init.zeros_(self.persistence[-1].bias)
        nn.init.constant_(self.action_persistence.weight,-3.)

    def export_config(self):
        return asdict(self.cfg)

    def observe(self, features):
        z = F.normalize(self.visual(features.float().flatten(-2)),dim=-1)
        return 12 * z @ F.normalize(self.observation_definitions,dim=-1).T

    def memory_sequence(self, observations, previous, hidden=None):
        x = torch.cat([observations,F.one_hot(previous,self.cfg.actions+1).float()],-1)
        states, hidden = self.memory(x.float(), hidden)
        logits = 12 * F.normalize(states,dim=-1) @ F.normalize(self.memory_definitions,dim=-1).T
        return logits, states, hidden

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
        logits = self.rule(inputs)
        lm,lo=logits.split([self.cfg.memories,self.cfg.observations],-1)
        gates=(self.persistence(inputs)+self.action_persistence(action)).sigmoid()
        pm=gates[...,0,None]*memory+(1-gates[...,0,None])*lm.softmax(-1)
        po=gates[...,1,None]*observation+(1-gates[...,1,None])*lo.softmax(-1)
        return pm.clamp_min(1e-9).log(),po.clamp_min(1e-9).log()

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

    @torch.no_grad()
    def compile(self, context, task):
        assert context.shape[0] == 1
        c=self.cfg
        s,o,a=torch.meshgrid(torch.arange(c.memories,device=context.device),
            torch.arange(c.observations,device=context.device),torch.arange(c.actions,device=context.device),indexing='ij')
        s=s.flatten();o=o.flatten();a=a.flatten();definitions=self.definitions(context)
        successors=[];confidences=[]
        for begin in range(0,len(s),4096):
            sm,ob=self.transition(context,F.one_hot(s[None,begin:begin+4096],c.memories).float(),
                F.one_hot(o[None,begin:begin+4096],c.observations).float(),a[None,begin:begin+4096],definitions)
            ms,mi=sm.softmax(-1).max(-1);os,oi=ob.softmax(-1).max(-1)
            successors.append((mi*c.observations+oi).squeeze(0).cpu())
            confidences.append((ms*os).squeeze(0).cpu())
        goal_m,goal_o=self.goal_logits(context,task)
        return dict(next_state=torch.cat(successors).reshape(c.memories*c.observations,c.actions).numpy(),
                    confidence=torch.cat(confidences).reshape(c.memories*c.observations,c.actions).numpy(),
                    goal=int(goal_m.argmax(-1))*c.observations+int(goal_o.argmax(-1)))
