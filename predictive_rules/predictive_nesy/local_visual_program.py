"""Boundary-only learned visual definitions, complete rules and a goal pointer.

Hypothetical actions expand learned visual state definitions at a revision only.
The expansion is independent of the selected goal and contains no spatial map.
All executable actions must subsequently come from the exact NumPy interpreter.
The proposal bank contains action strings from permitted training episodes;
there are no manually named predicates or hand-authored motion-state rules.
"""
from dataclasses import dataclass,asdict
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .discrete_rollout_world import DiscreteRolloutWorld
from .discrete_visual_world import WorldConfig
from .graph_subgoal_interpreter import Program

@dataclass
class LocalConfig:
    patches:int=49
    vocabulary:int=64
    actions:int=17
    horizon:int=10
    states:int=171
    hidden:int=128
    width:int=256

class GoalPointer(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.cfg=config or LocalConfig();c=self.cfg
        self.tokens=nn.Embedding(c.patches*c.vocabulary,8)
        self.register_buffer('offsets',torch.arange(c.patches)*c.vocabulary)
        self.observation=nn.Sequential(nn.Linear(c.patches*8,256),nn.GELU(),nn.Linear(256,c.width),nn.LayerNorm(c.width))
        self.context_read=nn.Sequential(nn.Linear(c.patches*8,c.hidden),nn.GELU(),nn.LayerNorm(c.hidden))
        self.action_embed=nn.Embedding(c.actions+1,c.hidden);self.task=nn.Embedding(2,c.hidden)
        self.history=nn.GRU(c.hidden,c.hidden,batch_first=True);self.context_norm=nn.LayerNorm(c.hidden)
        self.pair=nn.Sequential(nn.Linear(c.hidden+3*c.width,512),nn.GELU(),nn.Linear(512,128),nn.GELU(),nn.Linear(128,1))

    def embed(self,codes):return self.tokens(codes+self.offsets).flatten(-2)

    def forward(self,tasks,history,previous,lengths,definitions,valid):
        x=self.context_read(self.embed(history))+self.action_embed(previous)
        with torch.autocast(device_type=x.device.type,enabled=False):
            states,_=self.history(x.float(),self.task(tasks).float()[None])
        context=self.context_norm(states[torch.arange(len(tasks),device=tasks.device),lengths-1])
        current=self.observation(self.embed(history[torch.arange(len(tasks),device=tasks.device),lengths-1]))
        candidates=self.observation(self.embed(definitions));k=definitions.shape[1]
        features=torch.cat([context[:,None].expand(-1,k,-1),current[:,None].expand(-1,k,-1),candidates,candidates-current[:,None]],-1)
        return self.pair(features).squeeze(-1).float().masked_fill(~valid,-1e8)

class LocalVisualProgram(nn.Module):
    def __init__(self,world_config=None,proposals=None,config=None):
        super().__init__();self.cfg=config or LocalConfig();self.world=DiscreteRolloutWorld(world_config or WorldConfig())
        self.pointer=GoalPointer(self.cfg)
        if proposals is None:proposals=torch.zeros((1,10),dtype=torch.long)
        self.register_buffer('proposals',proposals.long())

    @torch.no_grad()
    def definitions(self,initial):
        """Only initial observations enter definition generation, never targets."""
        b=len(initial);sequences=[]
        for row in initial.detach().cpu().numpy():
            seed=int(hashlib.sha256(row.astype(np.uint8).tobytes()).hexdigest()[:16],16)
            rng=np.random.default_rng(seed);chosen=rng.integers(len(self.proposals),size=17)
            sampled=self.proposals[torch.tensor(chosen,device=initial.device)].clone()
            sampled[:,0]=torch.arange(17,device=initial.device)  # Exhaustive action IDs, no semantic state prior.
            sequences.append(sampled)
        actions=torch.stack(sequences).reshape(b*17,10)
        state=initial.repeat_interleave(17,0);outputs=[initial[:,None]]
        for t in range(10):
            state=self.world(state,actions[:,t]).argmax(-1)
            outputs.append(state.reshape(b,17,49))
        raw=torch.cat(outputs,1);definitions=[];valid=[];ground=[]
        for i in range(b):
            # Exact equal visual symbol vectors share an anonymous state.
            unique=torch.unique(raw[i],dim=0,sorted=True);n=len(unique)
            mask=torch.arange(self.cfg.states,device=initial.device)<n
            padded=torch.cat([unique,unique[:1].expand(self.cfg.states-n,-1)])
            definitions.append(padded);valid.append(mask)
            ground.append(torch.nonzero((unique==initial[i]).all(-1),as_tuple=False)[0,0])
        return torch.stack(definitions),torch.stack(valid),torch.stack(ground)

    @torch.no_grad()
    def rules(self,definitions,valid):
        """Generate every state/action row, without consulting goal or reachability."""
        tables=[]
        for codes,mask in zip(definitions,valid):
            k=len(codes);ids=torch.arange(17,device=codes.device).repeat(k)
            source=codes.repeat_interleave(17,0);blocks=[]
            targets=F.one_hot(codes,64).float().flatten(1).T.contiguous()
            for begin in range(0,len(source),512):
                logits=self.world(source[begin:begin+512],ids[begin:begin+512]).float().log_softmax(-1)
                likelihood=logits.flatten(1)@targets
                likelihood[:,~mask]=-1e8
                blocks.append(likelihood.softmax(-1))
            tables.append(torch.cat(blocks).reshape(k,17,k))
        return torch.stack(tables)

    @torch.no_grad()
    def generate(self,tasks,history,previous,lengths,*,start):
        if start<0 or start%10 or history.shape[1]>min(start+1,11):raise ValueError('Neural generation outside a revision boundary')
        initial=history[torch.arange(len(tasks),device=tasks.device),lengths-1]
        codes,valid,ground=self.definitions(initial)
        goal=self.pointer(tasks,history,previous,lengths,codes,valid)
        table=self.rules(codes,valid)
        return dict(definitions=codes,valid=valid,transitions=table,subgoal_logits=goal[:,None],ground=ground[:,None],start=start)

    def ground(self,program,codes):
        expected=program['definitions'][torch.arange(len(codes),device=codes.device),program['ground'][:,0]]
        if not torch.equal(expected,codes):raise ValueError('Grounding does not match boundary image')
        return F.one_hot(program['ground'],self.cfg.states).float()

def export_program(generated,index=0):
    if set(generated)!={'definitions','valid','transitions','subgoal_logits','ground','start'}:raise ValueError('Unexpected program field')
    n=int(generated['valid'][index].sum());goal=int(generated['subgoal_logits'][index,0].argmax())
    if not 0<=goal<n:raise ValueError('Invalid generated goal')
    return Program(generated['transitions'][index,:n,:,:n].detach().float().cpu().numpy().copy(),
        generated['definitions'][index,:n].detach().float().cpu().numpy().copy(),np.array([goal],np.int64),generated['start']).validate()

def label_distribution(definitions,valid,target):
    """Ground a future training-label image; never used in execution."""
    matches=(definitions==target[:,None]).float().sum(-1).masked_fill(~valid,-1.)
    best=(matches==matches.amax(-1,keepdim=True))&valid
    return best.float()/best.sum(-1,keepdim=True),matches.amax(-1)/definitions.shape[-1]
