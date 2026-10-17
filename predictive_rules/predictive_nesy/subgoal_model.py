"""Anonymous generated dynamics and a learned future symbolic subgoal."""
from dataclasses import dataclass,asdict
from itertools import combinations
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    patches:int=49
    vocabulary:int=64
    tasks:int=2
    factors:int=8
    states:int=16
    actions:int=17
    width:int=64
    hidden:int=128
    interval:int=10
    history:int=11
    observed_factors:int=4


def straight_through(p):
    return F.one_hot(p.argmax(-1),p.shape[-1]).to(p.dtype)+p-p.detach()


class RowDecoder(nn.Module):
    def __init__(self,sizes,hidden,outputs):
        super().__init__();self.sizes=tuple(sizes);self.outputs=outputs
        self.queries=nn.ModuleList(nn.Embedding(n,hidden) for n in sizes)
        self.context=nn.Linear(hidden,hidden)
        self.decoder=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,outputs))
        indices=torch.cartesian_prod(*(torch.arange(n) for n in sizes))
        self.register_buffer('indices',indices[:,None] if len(sizes)==1 else indices)

    def forward(self,context):
        q=sum(table(self.indices[:,i]) for i,table in enumerate(self.queries))
        return self.decoder(F.gelu(q[None]+self.context(context)[:,None])).reshape(len(context),*self.sizes,self.outputs)


class SubgoalRuleModel(nn.Module):
    execution_schema='subgoal_v1'
    def __init__(self, config=None):
        super().__init__();self.cfg=config or Config();c=self.cfg
        
        self.tokens=nn.Embedding(c.patches*c.vocabulary,8)
        self.register_buffer('offsets',torch.arange(c.patches)*c.vocabulary)
        self.observation=nn.Sequential(nn.Linear(c.patches*8,256),nn.GELU(),nn.Linear(256,c.observed_factors*c.width),nn.LayerNorm(c.observed_factors*c.width))
        self.memory_features=nn.Sequential(nn.Linear(c.hidden,256),nn.GELU(),nn.Linear(256,(c.factors-c.observed_factors)*c.width),nn.LayerNorm((c.factors-c.observed_factors)*c.width))
        self.context_read=nn.Sequential(nn.Linear(c.patches*8,c.hidden),nn.GELU(),nn.LayerNorm(c.hidden))
        self.action_embed=nn.Embedding(c.actions+1,c.hidden);self.task=nn.Embedding(c.tasks,c.hidden)
        self.history=nn.GRU(c.hidden,c.hidden,batch_first=True);self.context_norm=nn.LayerNorm(c.hidden)
        self.definitions=RowDecoder([c.factors,c.states],c.hidden,c.width)
        self.parents=RowDecoder([c.factors,2],c.hidden,c.factors)
        self.transitions=RowDecoder([c.factors,c.actions,c.states,c.states],c.hidden,c.states)
        self.subgoal=RowDecoder([c.factors],c.hidden,c.states)
        self.reconstruct=nn.Linear(c.factors*c.states,c.patches*c.vocabulary)
        self.inverse=nn.Sequential(nn.Linear(2*c.observed_factors*c.states,128),nn.GELU(),nn.Linear(128,c.actions))

    def export_config(self):return asdict(self.cfg)

    def embed(self,codes):
        if codes.dtype!=torch.long or codes.shape[-1]!=self.cfg.patches:raise ValueError('Invalid visual codes')
        return self.tokens(codes+self.offsets).flatten(-2)

    def generate(self,tasks,history,previous_actions,lengths,*,start):
        if start<0 or start%self.cfg.interval:raise ValueError('Not a rule-revision boundary')
        if history.ndim!=3 or history.shape[1]>min(start+1,self.cfg.history):raise ValueError('Noncausal history')
        if previous_actions.shape!=history.shape[:2]:raise ValueError('Missing preceding actions')
        inputs=self.context_read(self.embed(history))+self.action_embed(previous_actions)
        states,_=self.history(inputs,self.task(tasks)[None])
        raw=states[torch.arange(len(tasks),device=tasks.device),lengths-1]
        context=self.context_norm(raw)
        return dict(definitions=self.definitions(context),parents=self.parents(context).float().softmax(-1),
            transitions=self.transitions(context).float().softmax(-1),
            subgoal_logits=self.subgoal(context).float(),start=start,context_raw=raw)

    def ground_features(self,program,codes,memory):
        c=self.cfg
        visual=self.observation(self.embed(codes)).reshape(len(codes),c.observed_factors,c.width)
        history=self.memory_features(memory).reshape(len(codes),c.factors-c.observed_factors,c.width)
        features=torch.cat([visual,history],1)
        logits=torch.einsum('bfd,bfkd->bfk',F.normalize(features,dim=-1),F.normalize(program['definitions'],dim=-1))
        return (5*logits.float()).softmax(-1)

    def ground(self,program,codes):
        if codes.ndim!=2:raise ValueError('Boundary observation required')
        return self.ground_features(program,codes,program['context_raw'])

    def training_targets(self,program,codes,actions):
        memory=program['context_raw'];states=[self.ground_features(program,codes[:,0],memory)]
        for step in range(1,codes.shape[1]):
            x=self.context_read(self.embed(codes[:,step]))+self.action_embed(actions[:,step-1])
            _,h=self.history(x[:,None],memory[None]);memory=h[0]
            states.append(self.ground_features(program,codes[:,step],memory))
        return torch.stack(states,1)

