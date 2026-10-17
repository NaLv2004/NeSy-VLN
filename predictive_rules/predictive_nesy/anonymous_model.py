"""Anonymous categorical dynamics; no manually named semantic state heads."""
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


class AnonymousRuleModel(nn.Module):
    execution_schema='anonymous_v1'
    def __init__(self, config=None):
        super().__init__();self.cfg=config or Config();c=self.cfg
        self.pairs=list(combinations(range(c.factors),2));p=len(self.pairs)
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
        self.rewards=RowDecoder([p,c.actions,c.states,c.states],c.hidden,1)
        self.values=RowDecoder([p,c.states,c.states],c.hidden,1)
        self.rule_weights=nn.Linear(c.hidden,p)
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
            rewards=self.rewards(context).float().squeeze(-1).tanh(),
            values=self.values(context).float().squeeze(-1).tanh(),
            weights=self.rule_weights(context).float().softmax(-1),start=start,context_raw=raw)

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


def parent_joint(program,state):
    selectors=straight_through(program['parents'])
    marginal=torch.einsum('bfpg,b...gk->b...fpk',selectors,state)
    result=marginal[...,0,:,None]*marginal[...,1,None,:]
    same=selectors[:,:,0]*selectors[:,:,1]
    result=result-torch.einsum('bfg,b...gu,b...gv->b...fuv',same,state,state)
    return result+torch.diag_embed(torch.einsum('bfg,b...gk->b...fk',same,state))


def transition_all(program,state):
    return torch.einsum('b...fuv,bfauvk->b...afk',parent_joint(program,state),program['transitions'])


def scalar(program,state,name,pairs):
    left=state[...,[i for i,j in pairs],:];right=state[...,[j for i,j in pairs],:]
    rows=torch.einsum('b...pu,b...pv,bpuv->b...p',left,right,program[name])
    return (rows*program['weights'].reshape(len(state),*([1]*(rows.ndim-2)),-1)).sum(-1)


def reward_all(program,state,pairs,*,fixed_table=False):
    left=state[...,[i for i,j in pairs],:];right=state[...,[j for i,j in pairs],:]
    table=program['rewards'].detach() if fixed_table else program['rewards']
    weights=program['weights'].detach() if fixed_table else program['weights']
    rows=torch.einsum('b...pu,b...pv,bpauv->b...pa',left,right,table)
    return (rows*weights.reshape(len(state),*([1]*(rows.ndim-3)),-1,1)).sum(-2)


def scores(program,state,pairs):
    after=transition_all(program,state)
    continuation=.97*scalar(program,after,'values',pairs)
    continuation=torch.cat([torch.zeros_like(continuation[:,:1]),continuation[:,1:]],-1)
    return reward_all(program,state,pairs,fixed_table=True)+continuation
