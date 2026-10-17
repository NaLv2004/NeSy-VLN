"""Automatically learned, program-conditioned categorical navigation automata.

The only perceptual input is a frozen vocabulary of integer patch tokens.
Observation predicates, recurrent states and all rule tables are learned.
The interpreter uses categorical equality premises, not image features.
"""
from dataclasses import dataclass
from itertools import combinations
import torch
from torch import nn
import torch.nn.functional as F


def categorical(probabilities):
    codes=probabilities.argmax(-1)
    hard=F.one_hot(codes,probabilities.shape[-1]).to(probabilities)
    return codes,hard+(probabilities-probabilities.detach())


def categorical_usage_loss(probabilities,valid,hard_assignments=False):
    """Measure marginal symbol use without assigning any symbol a meaning.

Nearly uniform soft scores can still select one hard ID for every example.
The optional straight-through statistic measures the IDs actually consumed by
the interpreter. Padding has no influence on the statistic or its gradient.
"""
    if probabilities.ndim!=3 or valid.shape!=(len(probabilities),):
        raise ValueError('Expected batch x factors x categorical states')
    values=categorical(probabilities)[1] if hard_assignments else probabilities
    weights=valid.to(values)[:,None,None]
    mean=(values*weights).sum(0)/weights.sum().clamp_min(1)
    return (mean*(mean.clamp_min(1e-12)*probabilities.shape[-1]).log()).sum(-1).mean()


@dataclass
class AutomatonProgram:
    definitions: torch.Tensor
    transition_delta: torch.Tensor
    action_delta: torch.Tensor
    rule_weights: torch.Tensor
    coefficients: torch.Tensor
    start: int
    tasks: torch.Tensor | None = None


class FactorizedTokenAutomaton(nn.Module):
    """A small generator plus an explicit probabilistic rule interpreter.

Every ten actions the generator emits predicate definitions, transition
conclusions and action conclusions. Between revisions only integer states
persist. Straight-through training changes gradients, never forward values.
"""
    def __init__(self,patches=49,vocabulary=64,tasks=2,factors=4,states=16,
                 actions=17,width=64,rank=4,reset_memory=False,task_banks=False,context_bridge=False):
        super().__init__()
        self.patches=patches;self.vocabulary=vocabulary;self.tasks=tasks
        self.factors=factors;self.states=states;self.actions=actions
        self.width=width;self.rank=rank;self.reset_memory=reset_memory
        self.task_banks=bool(task_banks);self.context_bridge=bool(context_bridge)
        self.pairs=list(combinations(range(factors),2))
        if not self.pairs:raise ValueError('At least two independently learned state factors are required')
        self.tokens=nn.Embedding(patches*vocabulary,8)
        self.register_buffer('offsets',torch.arange(patches)*vocabulary)
        self.observation=nn.Sequential(nn.Linear(patches*8,256),nn.GELU(),
                                      nn.Linear(256,factors*width),nn.LayerNorm(factors*width))
        self.context_read=nn.Sequential(nn.Linear(patches*8,128),nn.GELU(),nn.LayerNorm(128))
        self.history=nn.GRU(128,128,batch_first=True)
        self.task=nn.Embedding(tasks,128)
        self.context=nn.Linear(128,rank)
        if self.context_bridge:
            self.bridge_norm=nn.LayerNorm(128)
            self.bridge=nn.Linear(256,rank)
            nn.init.zeros_(self.bridge.weight);nn.init.zeros_(self.bridge.bias)
        self.definitions=nn.Parameter(torch.randn(factors,states,width)*.1)
        self.definition_basis=nn.Parameter(torch.randn(rank,factors,states,width)*.01)
        # Premise: observation-factor=o AND prior-state=p AND prior-action=a.
        # Conclusion: a soft distribution over successor state IDs.
        self.transition=nn.Parameter(torch.randn(factors,states,states,actions+1,states)*.03)
        # Initially preserve arbitrary observation IDs, so recurrence does not
        # scramble an alphabet before it has learned any semantics. Every entry
        # remains trainable; simultaneous ID permutations preserve this prior.
        with torch.no_grad():
            for symbol in range(states):self.transition[:,symbol,:,:,symbol]+=3.
        self.transition_basis=nn.Parameter(torch.randn(rank,factors,states,states)*.01)
        # Premise: state_i=u AND state_j=v. All equality symbols are learned.
        self.conclusions=nn.Parameter(torch.randn(len(self.pairs),states,states,actions)*.03)
        self.conclusion_basis=nn.Parameter(torch.randn(rank,len(self.pairs),states,states,actions)*.01)
        self.weight_logits=nn.Parameter(torch.zeros(len(self.pairs)))
        self.weight_basis=nn.Parameter(torch.randn(rank,len(self.pairs))*.01)
        if self.task_banks:
            # Goal identity selects trainable program parameters. No state IDs,
            # predicate meanings, transitions or action choices are specified.
            for name in ['definitions','transition','conclusions']:
                value=getattr(self,name).detach()
                setattr(self,name,nn.Parameter(value[None].repeat(tasks,*([1]*value.ndim))))
        # Auxiliary objectives do not participate in action selection.
        self.reconstruct=nn.Linear(factors*states,patches*vocabulary)
        self.predict=nn.Sequential(nn.Linear(factors*states+actions,128),nn.GELU(),
                                   nn.Linear(128,patches*vocabulary))
        self.inverse=nn.Sequential(nn.Linear(2*factors*states,128),nn.GELU(),nn.Linear(128,actions))

    def validate_codes(self,codes):
        if codes.dtype!=torch.long or codes.shape[-1]!=self.patches or codes.ndim not in (2,3):
            raise TypeError('Only batched integer patch tokens may enter the automaton')
        if ((codes<0)|(codes>=self.vocabulary)).any():raise ValueError('Unknown learned visual token')

    def embed(self,codes):
        self.validate_codes(codes)
        return self.tokens(codes+self.offsets).flatten(-2)

    def generate(self,tasks,history,start):
        if type(start)is not int or start<0 or start%10:
            raise ValueError('Generate only at executed-action block boundaries')
        if history.ndim!=3 or not 1<=history.shape[1]<=min(start+1,11):
            raise ValueError('Program history must be causal and contain at most eleven observations')
        if tasks.dtype!=torch.long or tasks.shape!=(history.shape[0],) or ((tasks<0)|(tasks>=self.tasks)).any():
            raise TypeError('Expected one valid integer task token per trajectory')
        x=self.context_read(self.embed(history))
        _,h=self.history(x,self.task(tasks)[None])
        coefficients=.5*self.context(h[0]).tanh()
        if self.context_bridge:
            # Zero initialization preserves the old program exactly, while
            # allowing goal/history gradients even if the old tanh saturates.
            coefficients=coefficients+.1*self.bridge(torch.cat([self.bridge_norm(h[0]),self.bridge_norm(self.task(tasks))],-1))
        return self.program_from_coefficients(coefficients,start,tasks)

    def program_from_coefficients(self,c,start,tasks=None):
        if c.ndim!=2 or c.shape[-1]!=self.rank or not torch.isfinite(c).all():
            raise ValueError('Invalid generated program coefficients')
        if self.task_banks and (tasks is None or tasks.dtype!=torch.long or tasks.shape!=(len(c),) or ((tasks<0)|(tasks>=self.tasks)).any()):
            raise ValueError('A goal-conditioned program requires its original integer goal tokens')
        definitions=(self.definitions[tasks] if self.task_banks else self.definitions[None])+torch.einsum('br,rfkd->bfkd',c,self.definition_basis)
        transition=torch.einsum('br,rfok->bfok',c,self.transition_basis)
        actions=torch.einsum('br,rpuva->bpuva',c,self.conclusion_basis)
        weights=(self.weight_logits[None]+c@self.weight_basis).softmax(-1)
        return AutomatonProgram(definitions,transition,actions,weights,c,start,tasks)

    def initialize_from_shared(self,state):
        """Function-preserving transfer, not optimizer resume or new training."""
        target=self.state_dict();missing=[]
        for name,value in target.items():
            if name not in state:
                if name.startswith(('bridge.','bridge_norm.')):continue
                missing.append(name);continue
            source=state[name]
            if self.task_banks and name in ['definitions','transition','conclusions'] and source.ndim==value.ndim-1:
                source=source[None].repeat(self.tasks,*([1]*source.ndim))
            if source.shape!=value.shape:raise ValueError(f'Incompatible transferred tensor: {name}')
            target[name]=source
        if missing or set(state)-set(target):raise ValueError(f'Incomplete transfer: {missing}')
        self.load_state_dict(target)

    def initial_state(self,batch,device=None):
        # ID zero has no manually assigned meaning; the initial-action token
        # allows the learned transition to distinguish initialization.
        ids=torch.zeros(batch,self.factors,dtype=torch.long,device=device)
        return ids,F.one_hot(ids,self.states).float()

    def apply_action_rules(self,program,state_values,step):
        if not program.start<=step<program.start+10:raise ValueError('Expired program')
        if state_values.shape[-2:]!=(self.factors,self.states):raise ValueError('Invalid symbolic state')
        detached=state_values.detach()
        if not torch.equal(detached,detached.round()) or not torch.equal(detached.sum(-1),torch.ones_like(detached[...,0])):
            raise ValueError('Action rules require categorical states, never continuous visual latents')
        base=self.conclusions[program.tasks] if self.task_banks else self.conclusions[None]
        tables=(base+program.action_delta).softmax(-1)
        support=[]
        for r,(i,j) in enumerate(self.pairs):
            support.append(torch.einsum('bu,bv,buva->ba',state_values[:,i],state_values[:,j],tables[:,r]))
        evidence=torch.stack(support,1)
        probabilities=(program.rule_weights[:,:,None]*evidence).sum(1)
        return probabilities,evidence

    def step(self,program,codes,previous_values,previous_action,step):
        self.validate_codes(codes)
        if codes.ndim!=2:raise ValueError('One current symbolic observation is required')
        if previous_action.dtype!=torch.long or previous_action.shape!=(len(codes),) or ((previous_action<0)|(previous_action>self.actions)).any():
            raise TypeError('Invalid previous action tokens')
        if not program.start<=step<program.start+10:raise ValueError('Expired program')
        features=self.observation(self.embed(codes)).reshape(-1,self.factors,self.width)
        logits=torch.einsum('bfd,bfkd->bfk',F.normalize(features,dim=-1),F.normalize(program.definitions,dim=-1))*5.
        observed_prob=logits.softmax(-1);observed_ids,observed=categorical(observed_prob)
        if self.reset_memory:
            _,previous_values=self.initial_state(len(codes),codes.device)
            previous_action=torch.full_like(previous_action,self.actions)
        # Select a learned action-conditioned rule table. The complete table
        # exists as parameters; no continuous current observation enters it.
        tables=(self.transition[program.tasks,:,:,:,previous_action,:] if self.task_banks else
                self.transition[:,:,:,previous_action,:].permute(3,0,1,2,4))
        tables=tables+program.transition_delta[:,:,:,None,:]
        next_logits=torch.einsum('bfo,bfp,bfops->bfs',observed,previous_values,tables)
        state_prob=next_logits.softmax(-1);state_ids,state_values=categorical(state_prob)
        probabilities,evidence=self.apply_action_rules(program,state_values,step)
        return dict(probabilities=probabilities,rule_evidence=evidence,observed_ids=observed_ids,
                    observed_values=observed,observed_probabilities=observed_prob,state_ids=state_ids,
                    state_values=state_values,state_probabilities=state_prob,
                    reconstruction=self.reconstruct(observed.flatten(1)).reshape(-1,self.patches,self.vocabulary))

    def next_observation(self,state_values,executed_actions):
        action=F.one_hot(executed_actions,self.actions).to(state_values)
        return self.predict(torch.cat([state_values.flatten(1),action],-1)).reshape(-1,self.patches,self.vocabulary)

    def inverse_action(self,before,after):
        return self.inverse(torch.cat([before.flatten(1),after.flatten(1)],-1))

    @torch.no_grad()
    def proof(self,program,observed_ids,previous_ids,previous_action,state_ids,step):
        """Reproduce transitions and actions from saved IDs and program only."""
        for ids in (observed_ids,previous_ids,state_ids):
            if ids.dtype!=torch.long or ids.shape[-1]!=self.factors or ((ids<0)|(ids>=self.states)).any():
                raise TypeError('Proof requires learned integer state IDs')
        if self.reset_memory:
            previous_ids=torch.zeros_like(previous_ids);previous_action=torch.full_like(previous_action,self.actions)
        b=torch.arange(len(state_ids),device=state_ids.device)[:,None]
        f=torch.arange(self.factors,device=state_ids.device)[None,:]
        transition=(self.transition[program.tasks[:,None],f,observed_ids,previous_ids,previous_action[:,None]] if self.task_banks else
                    self.transition[f,observed_ids,previous_ids,previous_action[:,None]])
        transition=transition+program.transition_delta[b,f,observed_ids]
        if not torch.equal(transition.argmax(-1),state_ids):raise ValueError('Recorded state has no transition proof')
        onehot=F.one_hot(state_ids,self.states).float()
        p,evidence=self.apply_action_rules(program,onehot,step)
        return {'probabilities':p,'rule_evidence':evidence,'transition_probabilities':transition.softmax(-1),
                'pairs':self.pairs,'predicate_ids':observed_ids,'previous_states':previous_ids,'states':state_ids}
