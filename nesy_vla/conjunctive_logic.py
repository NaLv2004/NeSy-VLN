"""Generated soft conjunctions with learned true/false/unused conditions.

Each action rule consults an obligatory generated nonconstant literal and a
generated condition for every learned predicate. Nothing assigns human meaning
to a bit. Log-space evaluation avoids product underflow, not failed premises.
"""
import math,torch
from torch import nn
import torch.nn.functional as F
from .neural_logic import NeuralSymbolicLogic
from .symbols import DiscreteState
from .logic import NoSupportedAction


class ConjunctiveRuleLogic(NeuralSymbolicLogic):
    def __init__(self,predicates=32,actions=17,rules=96,neural_action_gates=False):
        super().__init__(predicates,actions,rules)
        if neural_action_gates:self.enable_action_gates()

    def enable_action_gates(self):
        if hasattr(self,'action_rule_gate'):raise ValueError('Action gates already exist')
        self.action_rule_gate=nn.Sequential(nn.Linear(self.k+72,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.action_rule_gate[-1].weight);nn.init.zeros_(self.action_rule_gate[-1].bias)

    def action_log_gates(self,p,states):
        if not isinstance(states,DiscreteState):raise TypeError('Rule gates require discrete predicate states')
        states.validate()
        if not p.neural_conjunction_gates:return states.values.new_zeros((len(states.values),self.action_slice.stop-self.action_slice.start))
        if not hasattr(self,'action_rule_gate'):raise ValueError('Generated program requires an installed neural rule interpreter')
        indices=torch.arange(self.action_slice.start,self.action_slice.stop,device=states.values.device)
        literal_ids=p.premises[self.action_slice].detach().argmax(-1)
        structure=torch.cat([self.literal_embedding(literal_ids).flatten(1),self.rule_embedding(indices)],-1)
        facts=states.values[:,None].expand(-1,len(indices),-1)
        inputs=torch.cat([facts,structure[None].expand(len(facts),-1,-1)],-1)
        logits=self.action_rule_gate(inputs).squeeze(-1)
        # Identity at initialization. The network emits rule weights, never
        # action logits; a false premise is still vetoed below.
        return F.logsigmoid(logits)+math.log(2.)

    def _proof(self,p,states):
        if not isinstance(states,DiscreteState):raise TypeError('Only discrete predicate states are allowed')
        states.validate()
        if states.values.ndim!=2 or states.values.shape[1]!=self.k:raise ValueError('Expected N by K states')
        if p.action_semantics!='conjunctive_rules' or p.conjunctive_conditions is None:
            raise ValueError('Generated conjunction conditions required')
        b=states.values.float();literals=torch.cat([b.new_ones((len(b),1)),b,1-b],-1)
        mandatory=literals@p.premises[self.action_slice,0].float().T
        conditions=p.conjunctive_conditions[self.action_slice].float()
        unused,positive,negative=conditions.unbind(-1)
        matched=unused[None]+positive[None]*b[:,None]+negative[None]*(1-b[:,None])
        weights=p.weights[self.action_slice].float()[None]
        active=(mandatory>0)&(matched>0).all(-1)&(weights>0)
        if p.conjunction_operator=='product':
            log_truth=mandatory.clamp_min(1e-30).log()+matched.clamp_min(1e-30).log().sum(-1)
        elif p.conjunction_operator=='minimum':
            # Godel conjunction retains ordinary Boolean AND and zero support
            # for a false premise, without multiplying uncertain soft premises.
            log_truth=torch.minimum(mandatory,matched.amin(-1)).clamp_min(1e-30).log()
        else:raise ValueError('Unknown conjunction operator')
        log_gates=self.action_log_gates(p,states)
        log_mass=log_truth+weights.clamp_min(1e-30).log()+log_gates
        log_mass=log_mass.masked_fill(~active,-torch.inf)
        if not torch.isfinite(log_mass).any(-1).all():raise NoSupportedAction('No generated conjunction is satisfied')
        mass=log_mass.softmax(-1);contribution=mass[:,:,None]*p.action_heads[self.action_slice].float()[None]
        probabilities=contribution.sum(1)
        return probabilities,contribution,mass,log_mass,matched,mandatory,log_gates

    def probabilities_for_states(self,p,states):return self._proof(p,states)[0]

    def infer(self,p,state,confidence=None,disabled_rule=None):
        if not isinstance(state,DiscreteState):raise TypeError('Only discrete predicate states are allowed')
        state.validate()
        if state.values.ndim!=1:raise ValueError('One actual state required')
        if disabled_rule is not None:
            mask=torch.ones_like(p.weights);mask[disabled_rule]=0
            p=p.map(lambda value:value);p.weights=p.weights*mask
        batch=DiscreteState(state.codes[None],state.values[None])
        probabilities,contribution,mass,logs,matched,mandatory,log_gates=self._proof(p,batch)
        return {'probs':probabilities[0],'support':probabilities[0],
            'belief':state.values,'confidence':torch.ones_like(state.values),
            'action_activation':mass[0],'action_contribution':contribution[0],
            'layers':[{'mandatory_match':mandatory[0],'predicate_matches':matched[0],
                       'log_rule_mass':logs[0].masked_fill(~torch.isfinite(logs[0]),-1e30),
                       'applicable':torch.isfinite(logs[0]).float()}],
            'symbol_codes':state.codes,'symbol_values':state.values,'rule_gates':log_gates[0].exp()}
