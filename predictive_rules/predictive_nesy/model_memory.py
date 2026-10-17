"""Generated categorical world model with learned predictive memory factors.

The old observation labels remain auxiliary anchors for four visual factors.
Four additional factors are learned from causal observation/action histories.
Their category meanings and transitions are never manually assigned.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .model import ModelConfig, GeneratedWorldModel, straight_through


@dataclass
class MemoryConfig(ModelConfig):
    factors: int = 8
    observed_factors: int = 4


class PredictiveMemoryModel(GeneratedWorldModel):
    def __init__(self, config=None):
        super().__init__(config or MemoryConfig())
        c=self.cfg
        self.observation=nn.Sequential(nn.Linear(c.patches*8,256),nn.GELU(),
            nn.Linear(256,c.observed_factors*c.width),nn.LayerNorm(c.observed_factors*c.width))
        self.memory_features=nn.Sequential(nn.Linear(c.hidden,256),nn.GELU(),
            nn.Linear(256,(c.factors-c.observed_factors)*c.width),nn.LayerNorm((c.factors-c.observed_factors)*c.width))
        # The inverse head cannot see action-bearing memory: otherwise its label
        # would already be encoded in its input.
        self.inverse=nn.Sequential(nn.Linear(2*c.observed_factors*c.states,128),nn.GELU(),nn.Linear(128,c.actions))

    def generate(self,tasks,history,previous_actions,lengths,*,start):
        if start<0 or start%self.cfg.interval:
            raise ValueError('Neural program generation requires a rule boundary')
        if history.ndim!=3 or history.shape[1]>min(start+1,self.cfg.history):
            raise ValueError('Noncausal boundary history')
        if previous_actions.shape!=history.shape[:2]:
            raise ValueError('Missing preceding action IDs')
        inputs=self.context_read(self.embed(history))+self.action_embed(previous_actions)
        states,_=self.history(inputs,self.task(tasks)[None])
        raw=states[torch.arange(len(tasks),device=tasks.device),lengths-1]
        context=self.context_norm(raw)
        return dict(definitions=self.definitions(context),parents=self.parents(context).float().softmax(-1),
            transitions=self.transitions(context).float().softmax(-1),risks=self.risks(context).float().squeeze(-1).sigmoid(),
            goals=self.goals(context).float().squeeze(-1).sigmoid(),values=self.values(context).float().squeeze(-1).sigmoid(),
            weights=self.rule_weights(context).float().softmax(-1),start=start,context_raw=raw)

    def ground_features(self,program,codes,memory):
        c=self.cfg
        visual=self.observation(self.embed(codes)).reshape(len(codes),c.observed_factors,c.width)
        history=self.memory_features(memory).reshape(len(codes),c.factors-c.observed_factors,c.width)
        features=torch.cat([visual,history],1)
        logits=torch.einsum('bfd,bfkd->bfk',F.normalize(features,dim=-1),F.normalize(program['definitions'],dim=-1))
        return (5*logits.float()).softmax(-1)

    def ground(self,program,codes):
        if codes.ndim!=2:
            raise ValueError('Deployment grounding consumes the current boundary image only')
        return self.ground_features(program,codes,program['context_raw'])

    def training_targets(self,program,codes,executed_actions):
        """Future encodings are labels. They are not fed to rule rollouts."""
        memory=program['context_raw']
        values=[self.ground_features(program,codes[:,0],memory)]
        for step in range(1,codes.shape[1]):
            features=self.context_read(self.embed(codes[:,step]))+self.action_embed(executed_actions[:,step-1])
            _,h=self.history(features[:,None],memory[None]);memory=h[0]
            values.append(self.ground_features(program,codes[:,step],memory))
        return torch.stack(values,1)


def parent_joint(program,state):
    """Marginalize categorical premises, retaining same-variable correlation.

    Distinct factors use the explicit factorized-belief approximation. Two
    references to the SAME factor use diag(p), never the incorrect p outer p.
    """
    selectors=straight_through(program['parents'])
    marginals=torch.einsum('bfpg,bgk->bfpk',selectors,state)
    joint=marginals[:,:,0,:,None]*marginals[:,:,1,None,:]
    same=selectors[:,:,0]*selectors[:,:,1]
    joint=joint-torch.einsum('bfg,bgu,bgv->bfuv',same,state,state)
    joint=joint+torch.diag_embed(torch.einsum('bfg,bgk->bfk',same,state))
    return joint


def transition_all(program,state):
    return torch.einsum('bfuv,bfauvk->bafk',parent_joint(program,state),program['transitions'])


def risk_all(program,state):
    return torch.einsum('bfuv,bfauv->ba',parent_joint(program,state),program['risks'])/state.shape[-2]


def scalar(program,state,name,pairs):
    first=state[...,[i for i,j in pairs],:]
    second=state[...,[j for i,j in pairs],:]
    values=torch.einsum('b...pu,b...pv,bpuv->b...p',first,second,program[name])
    weights=program['weights'].reshape(len(state),*([1]*(values.ndim-2)),-1)
    return (values*weights).sum(-1)


def scores(program,state,pairs,gamma=.97,step_cost=.02,collision_cost=.2):
    after=transition_all(program,state)
    value=scalar(program,after,'values',pairs)
    q=gamma*value-step_cost-collision_cost*risk_all(program,state).detach()
    stop=2*scalar(program,state,'goals',pairs)-1
    return torch.cat([stop[:,None],q[:,1:]],-1)
