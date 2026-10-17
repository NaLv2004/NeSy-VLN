"""Generated anonymous finite-state dynamics and a learned discrete subgoal."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F
from .subgoal_model import RowDecoder


class _HardCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx,probabilities):return F.one_hot(probabilities.argmax(-1),probabilities.shape[-1]).to(probabilities.dtype)
    @staticmethod
    def backward(ctx,gradient):return gradient


def straight_through(probabilities):return _HardCategorical.apply(probabilities)


@dataclass
class Config:
    patches:int=49
    vocabulary:int=64
    tasks:int=2
    states:int=128
    actions:int=17
    width:int=256
    hidden:int=128
    decoder_hidden:int=512
    interval:int=10
    history:int=11


class WideRowDecoder(nn.Module):
    def __init__(self,sizes,context_width,hidden,outputs):
        super().__init__();self.sizes=tuple(sizes);self.outputs=outputs
        self.queries=nn.ModuleList(nn.Embedding(n,hidden) for n in sizes)
        self.context=nn.Linear(context_width,hidden)
        self.decoder=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,outputs))
        indices=torch.cartesian_prod(*(torch.arange(n) for n in sizes))
        self.register_buffer('indices',indices[:,None] if len(sizes)==1 else indices)
    def forward(self,context):
        query=sum(table(self.indices[:,i]) for i,table in enumerate(self.queries))
        return self.decoder(F.gelu(query[None]+self.context(context)[:,None])).reshape(len(context),*self.sizes,self.outputs)


class GraphSubgoalModel(nn.Module):
    execution_schema='graph_subgoal_v1'
    def __init__(self,config=None):
        super().__init__();self.cfg=config or Config();c=self.cfg
        self.tokens=nn.Embedding(c.patches*c.vocabulary,8);self.register_buffer('offsets',torch.arange(c.patches)*c.vocabulary)
        self.observation=nn.Sequential(nn.Linear(c.patches*8,256),nn.GELU(),nn.Linear(256,c.width),nn.LayerNorm(c.width))
        self.memory_features=nn.Sequential(nn.Linear(c.hidden,256),nn.GELU(),nn.Linear(256,c.width),nn.LayerNorm(c.width))
        self.fuse=nn.Sequential(nn.Linear(2*c.width,c.width),nn.GELU(),nn.LayerNorm(c.width))
        self.context_read=nn.Sequential(nn.Linear(c.patches*8,c.hidden),nn.GELU(),nn.LayerNorm(c.hidden))
        self.action_embed=nn.Embedding(c.actions+1,c.hidden);self.task=nn.Embedding(c.tasks,c.hidden)
        self.history=nn.GRU(c.hidden,c.hidden,batch_first=True);self.context_norm=nn.LayerNorm(c.hidden)
        self.definitions=WideRowDecoder([c.states],c.hidden,c.decoder_hidden,c.width)
        self.transitions=WideRowDecoder([c.states,c.actions],c.hidden,c.decoder_hidden,c.states)
        self.subgoal=nn.Linear(c.hidden,c.states)
        self.reconstruct=nn.Sequential(nn.Linear(c.states+c.hidden,256),nn.GELU(),nn.Linear(256,c.patches*c.vocabulary))

    def export_config(self):return asdict(self.cfg)
    def embed(self,codes):
        if codes.dtype!=torch.long or codes.shape[-1]!=self.cfg.patches:raise ValueError('Invalid visual codes')
        return self.tokens(codes+self.offsets).flatten(-2)

    def generate(self,tasks,history,previous_actions,lengths,*,start):
        if start<0 or start%10 or history.shape[1]>min(start+1,11):raise ValueError('Noncausal generation')
        x=self.context_read(self.embed(history))+self.action_embed(previous_actions)
        # cuDNN GRU can execute in FP16 even under a BF16 autocast context.
        # Long discrete-rollout gradients then overflow before gradient clipping.
        # Keep only this small recurrent block in FP32; rule decoders retain AMP.
        with torch.autocast(device_type=x.device.type,enabled=False):
            states,_=self.history(x.float(),self.task(tasks).float()[None])
        raw=states[torch.arange(len(tasks),device=tasks.device),lengths-1]
        context=self.context_norm(raw)
        return dict(definitions=self.definitions(context),transitions=self.transitions(context).float().softmax(-1),
            subgoal_logits=self.subgoal(context).float()[:,None],context_raw=raw,start=start)

    def ground_features(self,program,codes,memory):
        feature=self.fuse(torch.cat([self.observation(self.embed(codes)),self.memory_features(memory)],-1))
        logits=torch.einsum('bd,bkd->bk',F.normalize(feature,dim=-1),F.normalize(program['definitions'],dim=-1))
        return (8*logits.float()).softmax(-1)[:,None]

    def ground(self,program,codes):return self.ground_features(program,codes,program['context_raw'])

    def training_targets(self,program,codes,actions):
        memory=program['context_raw'];states=[self.ground_features(program,codes[:,0],memory)]
        for i in range(1,codes.shape[1]):
            x=self.context_read(self.embed(codes[:,i]))+self.action_embed(actions[:,i-1])
            _,h=self.history(x[:,None],memory[None]);memory=h[0]
            states.append(self.ground_features(program,codes[:,i],memory))
        return torch.stack(states,1)

    def reconstruct_codes(self,program,states):
        context=self.context_norm(program['context_raw'])[:,None].expand(-1,states.shape[1],-1)
        return self.reconstruct(torch.cat([states,context],-1)).reshape(*states.shape[:2],self.cfg.patches,self.cfg.vocabulary)
