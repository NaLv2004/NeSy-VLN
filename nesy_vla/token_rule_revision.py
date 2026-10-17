"""Low-frequency program revision over automatically discovered token facts.

The reviser returns rule-program weights, never action scores. Current actions
can only be obtained by replaying the selected categorical conjunction trees.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class TokenFacts:
    codes: torch.Tensor
    cardinalities: tuple

    def validate(self):
        if self.codes.dtype != torch.long or self.codes.ndim not in (1,2,3):
            raise TypeError('Only integer symbolic facts may cross the interpreter boundary')
        if self.codes.shape[-1] != len(self.cardinalities):
            raise ValueError('Symbol schema differs')
        limits=self.codes.new_tensor(self.cardinalities)
        if any(type(n) is not int or n<1 for n in self.cardinalities):
            raise ValueError('Invalid categorical symbol vocabulary')
        if ((self.codes<0)|(self.codes>=limits)).any():
            raise ValueError('Unknown symbolic value')
        return self


@dataclass
class RuleRevision:
    weights: torch.Tensor
    start_step: int
    interval: int

    def validate(self,trees):
        if self.weights.ndim!=1 or self.weights.numel()!=trees:
            raise ValueError('Revision must assign one weight to each generated tree')
        if not torch.isfinite(self.weights).all() or (self.weights<0).any():
            raise ValueError('Invalid rule weights')
        if not torch.isclose(self.weights.sum(),self.weights.new_tensor(1.),atol=1e-6):
            raise ValueError('Rule weights must sum to one')
        if type(self.start_step)is not int or self.start_step<0 or self.interval!=10:
            raise ValueError('Program revision follows executed ten-action blocks')
        return self


class TokenProgramReviser(nn.Module):
    """A small learned program generator with no access to image tensors.

All categorical inputs and their cardinalities are learned tokens or past
action IDs. Vocabulary IDs and patch addresses assign no semantic meanings.
The returned program remains fixed for the next ten executed actions.
"""
    def __init__(self,cardinalities,tasks,trees,embedding_dim=8,hidden=96,detach_value_features=False):
        super().__init__();self.cardinalities=tuple(cardinalities);self.trees=trees;self.tasks=tasks
        self.detach_value_features=bool(detach_value_features)
        offsets=[0]
        for n in cardinalities:offsets.append(offsets[-1]+n)
        self.register_buffer('offsets',torch.tensor(offsets[:-1],dtype=torch.long))
        self.symbols=nn.Embedding(offsets[-1],embedding_dim)
        self.task=nn.Embedding(tasks,hidden)
        self.read=nn.Sequential(nn.Linear(len(cardinalities)*embedding_dim,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.history=nn.GRU(hidden,hidden,batch_first=True)
        self.program=nn.Linear(hidden,trees)
        self.critic=nn.Linear(hidden,1)
        # The first program exactly reproduces equal weighting of induced rules.
        nn.init.zeros_(self.program.weight);nn.init.zeros_(self.program.bias)
        nn.init.zeros_(self.critic.weight);nn.init.zeros_(self.critic.bias)

    def forward(self,task,history,start_step):
        return self.generate(task,history,start_step)[0]

    def generate(self,task,history,start_step):
        if not isinstance(history,TokenFacts):raise TypeError('Reviser accepts TokenFacts only')
        history.validate()
        if history.cardinalities!=self.cardinalities or history.codes.ndim!=2:
            raise ValueError('Expected one chronological token sequence')
        if type(task)is not int or not 0<=task<self.tasks:raise ValueError('Unknown task token')
        if type(start_step)is not int or start_step<0 or start_step%10:
            raise ValueError('Only regenerate at executed-action block boundaries')
        if not 1<=len(history.codes)<=min(start_step+1,11):
            raise ValueError('History length is not causal for this revision boundary')
        embedded=self.symbols(history.codes+self.offsets).flatten(1)
        values=self.read(embedded)[None]
        initial=self.task.weight[task][None,None]
        _,hidden=self.history(values,initial)
        revision=RuleRevision(self.program(hidden[0,0]).softmax(-1),start_step,10)
        value_features=hidden[0,0].detach() if self.detach_value_features else hidden[0,0]
        return revision.validate(self.trees),self.critic(value_features).squeeze(-1)


def replay_tree_program(trees,facts,revision,step):
    """Strict Boolean path satisfaction; weighted conclusions remain soft.

There is no fallback. Changing weights cannot enable a leaf with a false
premise. Inputs include no RGB, real-valued visual latents, poses or map.
"""
    if not isinstance(facts,TokenFacts):raise TypeError('Only token facts may enter reasoning')
    facts.validate();revision.validate(len(trees))
    if facts.codes.ndim!=1:raise ValueError('Expected one current symbolic state')
    if type(step)is not int or not revision.start_step<=step<revision.start_step+revision.interval:
        raise ValueError('Missing or expired generated program')
    evidence=[];certificates=[]
    for tree_index,tree in enumerate(trees):
        node=0;clauses=[];seen=set()
        while tree['slots'][node]>=0:
            if node in seen:raise ValueError('Cyclic proof tree')
            seen.add(node);slot=tree['slots'][node];value=tree['values'][node]
            if not 0<=slot<len(facts.cardinalities) or not 0<=value<facts.cardinalities[slot]:
                raise ValueError('Rule literal is outside its learned symbol vocabulary')
            truth=bool(facts.codes[slot]==value);clauses.append((slot,value,truth))
            node=tree['true_branch' if truth else 'false_branch'][node]
            if not 0<=node<len(tree['slots']):raise ValueError('Invalid rule branch')
        conclusion=revision.weights.new_tensor(tree['conclusion'][node])
        if conclusion.ndim!=1 or not torch.isfinite(conclusion).all() or (conclusion<0).any():
            raise ValueError('Invalid generated conclusion')
        if not torch.isclose(conclusion.sum(),conclusion.new_tensor(1.),atol=1e-6):
            raise ValueError('Leaf conclusion must be normalized')
        evidence.append(conclusion);certificates.append({'tree':tree_index,'leaf':node,'clauses':clauses})
    evidence=torch.stack(evidence)
    probabilities=(revision.weights[:,None]*evidence).sum(0)
    return probabilities,certificates,evidence


def convert_v43_trees(trees,cardinalities,stride=65):
    """Decode stored automatically induced one-hot feature addresses."""
    converted=[]
    for tree in trees:
        slots=[];values=[]
        for feature in tree['predicate']:
            if feature<0:slots.append(-1);values.append(-1)
            else:
                slot,value=divmod(feature,stride)
                if slot>=len(cardinalities) or value>=cardinalities[slot]:
                    raise ValueError('Induced rule refers to unknown categorical fact')
                slots.append(slot);values.append(value)
        converted.append(dict(slots=slots,values=values,false_branch=tree['false_branch'],
                              true_branch=tree['true_branch'],conclusion=tree['conclusion']))
    return converted


class PackedRuleProgram:
    """Vectorized execution of the same learned Boolean decision trees."""
    def __init__(self,trees,cardinalities):
        import numpy as np
        self.cardinalities=tuple(cardinalities);self.roots=[]
        slots=[];values=[];left=[];right=[];conclusions=[]
        for tree in trees:
            start=len(slots);self.roots.append(start)
            actions=len(next(x for x in tree['conclusion'] if x is not None))
            for node,slot in enumerate(tree['slots']):
                slots.append(slot);values.append(tree['values'][node])
                left.append(start+tree['false_branch'][node]);right.append(start+tree['true_branch'][node])
                conclusions.append(tree['conclusion'][node] if slot<0 else [0.]*actions)
        self.roots=np.array(self.roots);self.slots=np.array(slots);self.values=np.array(values)
        self.left=np.array(left);self.right=np.array(right);self.conclusions=np.asarray(conclusions)

    def evidence(self,facts):
        import numpy as np
        if not isinstance(facts,TokenFacts):raise TypeError('Packed proof requires integer facts')
        facts.validate()
        if facts.codes.ndim!=2 or facts.cardinalities!=self.cardinalities:raise ValueError('Expected batched schema-matched token facts')
        states=facts.codes.detach().cpu().numpy();nodes=np.tile(self.roots,(len(states),1));visited=0
        while (self.slots[nodes]>=0).any():
            active=self.slots[nodes]>=0;slots=self.slots[nodes].clip(0)
            truth=np.take_along_axis(states,slots,axis=1)==self.values[nodes]
            next_node=np.where(truth,self.right[nodes],self.left[nodes])
            nodes=np.where(active,next_node,nodes);visited+=1
            if visited>len(self.slots):raise ValueError('Cyclic or invalid generated program')
        return self.conclusions[nodes],nodes
