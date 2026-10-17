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
        self.memory = nn.GRUCell(c.observations+c.actions+1+c.width,c.width)
        # An independently parameterized predictor sees CURRENT premises only.
        # The perception-side memory encoder instead grounds labels in actual
        # observed histories and is called at runtime only during rule renewal.
        self.memory_predictor = nn.GRUCell(c.observations+c.actions+1,c.width)
        self.frame = nn.Linear(49*128,c.context)
        self.context_history = nn.GRU(c.context+c.observations+c.actions+1,c.context,batch_first=True)
        self.task = nn.Embedding(2,c.context)
        self.definition_generator = nn.Linear(c.context,c.memories*c.width)
        self.observation_embedding = nn.Linear(c.observations,c.width,bias=False)
        self.action_embedding = nn.Embedding(c.actions,c.width)
        self.rule = nn.Sequential(nn.Linear(c.width*3+c.context,256),nn.SiLU(),nn.Linear(256,c.observations))
        self.memory_rule = nn.Sequential(nn.Linear(c.width*3+c.context,128),nn.SiLU(),nn.Linear(128,c.width))
        nn.init.zeros_(self.memory_rule[-1].weight);nn.init.zeros_(self.memory_rule[-1].bias)
        self.goal = nn.Sequential(nn.Linear(c.context,256),nn.SiLU(),nn.Linear(256,c.memories+c.observations))
        self.renewal = nn.Sequential(nn.Linear(c.memories+c.observations*2+1,128),nn.SiLU(),nn.Linear(128,1))
        # Auxiliary heads train abstractions, never select runtime actions.
        # Observation codes ONLY. A future memory code contains action history
        # and would leak the answer into inverse prediction.
        self.inverse = nn.Sequential(nn.Linear(2*c.observations+10,256),nn.SiLU(),nn.Linear(256,c.actions))
        self.reconstruct = nn.Linear(c.observations,49*128)
        self.memory_reconstruct = nn.Linear(c.memories,c.width)
        # Training-only joint reconstruction grounds anonymous codes in images.
        self.state_reconstruct = nn.Sequential(nn.Linear(c.memories+c.observations,256),
            nn.SiLU(),nn.Linear(256,49*128))
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
        # Entire 64x128 observation-definition vectors are generated at the
        # rule boundary. They are fixed for the lifetime of that program.
        self.observation_definition_generator=nn.Linear(c.context,c.observations*c.width)
        nn.init.zeros_(self.observation_definition_generator.weight)
        nn.init.zeros_(self.observation_definition_generator.bias)
        self.definition_feature_decoder=nn.Linear(c.width,49*128,bias=False)
        nn.init.normal_(self.definition_feature_decoder.weight,std=.01)
        self.detach_copy_evidence=True
        self.detach_memory_carrier=True

    def export_config(self):
        return asdict(self.cfg)

    def observation_definition_delta(self, context):
        return self.observation_definition_generator(context).reshape(-1,self.cfg.observations,self.cfg.width)

    def observation_encoding(self, features, context=None):
        z=F.normalize(self.visual(features.float().flatten(-2)),dim=-1)
        if context is None:
            definitions=F.normalize(self.observation_definitions,dim=-1)
            logits=12*z@definitions.T
            selected=definitions[logits.detach().argmax(-1)]
        else:
            assert len(features)==len(context)
            definitions=F.normalize(self.observation_definitions[None]+self.observation_definition_delta(context),dim=-1)
            logits=12*torch.einsum('b...w,bow->b...o',z,definitions)
            discrete=F.one_hot(logits.detach().argmax(-1),self.cfg.observations).float()
            selected=torch.einsum('b...o,bow->b...w',discrete,definitions)
        return logits,z,selected

    def observe(self, features, context=None):
        return self.observation_encoding(features,context)[0]

    def definition_residual(self, observation, context):
        delta=torch.einsum('b...o,bow->b...w',observation,self.observation_definition_delta(context))
        return self.definition_feature_decoder(delta)

    def load_v43(self, state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected
        assert set(missing)=={'observation_definition_generator.weight',
            'observation_definition_generator.bias','definition_feature_decoder.weight'}

    def memory_step(self, memory, observation, previous, evidence, initial=False):
        """Perception-side grounding from previous memory and arriving evidence.

        The zero initial carrier is a fixed sequence boundary, not a hidden
        recurrent history. This encoder supplies labels during training and
        a grounded initial state at a runtime rule boundary. It is NOT the
        separately parameterized future-memory predictor used by world rules.
        """
        shape=observation.shape[:-1]
        assert evidence.shape == (*observation.shape[:-1],self.cfg.width)
        x=torch.cat([observation,F.one_hot(previous,self.cfg.actions+1).float(),evidence],-1)
        carrier=memory.detach() if memory is not None and self.detach_memory_carrier else memory
        hidden=(torch.zeros(*shape,self.cfg.width,device=x.device) if initial
                else self.memory_reconstruct(carrier))
        state=self.memory(x.reshape(-1,x.shape[-1]),hidden.reshape(-1,self.cfg.width)).reshape(*shape,self.cfg.width)
        logits=12*F.normalize(state,dim=-1)@F.normalize(self.memory_definitions,dim=-1).T
        return logits,state

    def memory_sequence(self, observations, previous, evidence, hidden=None):
        # `hidden`, if supplied for a counterfactual successor, is a hard memory
        # distribution of width memories, NOT the GRU's continuous activation.
        carrier=hidden[0] if hidden is not None else None
        logits=[];states=[]
        for t in range(observations.shape[1]):
            # This is a grounded HISTORY ENCODER, not runtime state execution.
            # Actual O_t may ground M_t during training and at rule boundaries.
            # Between boundaries, the interpreter uses exported predictions;
            # it never calls this encoder to replace its ongoing hard memory.
            source_observation=observations[:,t]
            lm,state=self.memory_step(carrier,source_observation,previous[:,t],evidence[:,t],initial=carrier is None)
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
        # Future memory is predicted now from current hard premises. The
        # separate supervision encoder can see recorded future evidence during
        # training; this predictor and compiled table cannot see that evidence.
        carrier=memory.detach() if self.detach_memory_carrier else memory
        hidden=self.memory_reconstruct(carrier)
        current_inputs=torch.cat([observation,F.one_hot(action,self.cfg.actions+1).float()],-1)
        state=self.memory_predictor(current_inputs.reshape(-1,current_inputs.shape[-1]),
            hidden.reshape(-1,self.cfg.width)).reshape(*memory.shape[:-1],self.cfg.width)
        state=state+self.memory_rule(inputs)
        lm=12*F.normalize(state,dim=-1)@F.normalize(self.memory_definitions,dim=-1).T
        return lm,log_observation

    @torch.no_grad()
    def initialize_memory_predictor(self):
        """Initialize added parameters without tying them to the label encoder."""
        state=dict(self.memory.state_dict())
        state['weight_ih']=state['weight_ih'][:,:self.cfg.observations+self.cfg.actions+1].clone()
        self.memory_predictor.load_state_dict(state)
        self.memory_rule[-1].weight.zero_();self.memory_rule[-1].bias.zero_()

    def initialize_v41(self, state):
        """Reuse old weights while separating predictor and grounding encoder."""
        state=dict(state)
        old=state['memory.weight_ih']
        weight=self.memory.weight_ih.detach().clone()
        weight[:,:old.shape[1]]=old
        weight[:,old.shape[1]:]*=.2
        state['memory.weight_ih']=weight
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected
        assert all(k.startswith(('memory_predictor.','memory_rule.','state_reconstruct.')) for k in missing),missing
        self.initialize_memory_predictor()

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
