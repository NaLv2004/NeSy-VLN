"""Learned rule weighting over discrete facts, with explicit rule execution.

The network sees Boolean predicate tokens and structural rule IDs only. It can
weight an applicable generated rule, but cannot make a false premise fire or
emit a motor action. Actions still come from the generated rule conclusions.
"""
import torch
from torch import nn
from .logic import SoftLogic
from .symbols import DiscreteState,tokenize


class NeuralSymbolicLogic(nn.Module,SoftLogic):
    def __init__(self,predicates=32,actions=17,rules=96):
        nn.Module.__init__(self);SoftLogic.__init__(self,predicates,actions,rules)
        self.literal_embedding=nn.Embedding(2*predicates+1,24)
        self.rule_embedding=nn.Embedding(rules,24)
        self.rule_gate=nn.Sequential(nn.Linear(predicates+72,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.rule_gate[-1].weight);nn.init.constant_(self.rule_gate[-1].bias,2.)
        self.state_update=nn.Sequential(nn.Linear(2*predicates,64),nn.GELU(),nn.Linear(64,predicates))
        nn.init.zeros_(self.state_update[-1].weight);nn.init.constant_(self.state_update[-1].bias,1.)

    def assimilate(self,observed,predicted=None):
        """Update the automaton valuation using discrete observation and memory.

        The previous value is produced by the generated action-effect rules.
        A learned symbolic selector decides which conflicting bits to retain;
        no image features, visual confidences or geometric state are accepted.
        """
        if not isinstance(observed,DiscreteState):raise TypeError('Expected observed symbol tokens')
        observed.validate()
        if predicted is None:return observed
        if not isinstance(predicted,DiscreteState):raise TypeError('Expected predicted symbol tokens')
        predicted.validate()
        gate=self.state_update(torch.cat([observed.values,predicted.values])).sigmoid()
        return tokenize(gate*observed.values+(1-gate)*predicted.values)

    def gates(self,p,state,subset):
        if not isinstance(state,DiscreteState):raise TypeError('Raw features are forbidden in rule weighting')
        state.validate()
        return self._validated_gates(p,state,subset)

    def _validated_gates(self,p,state,subset):
        indices=torch.arange(subset.start,subset.stop,device=state.values.device)
        # These are vocabulary indices of predicates in the generated program,
        # not image embeddings or manually named navigation predicates.
        literals=p.premises[subset].detach().argmax(-1)
        structure=torch.cat([self.literal_embedding(literals).flatten(1),self.rule_embedding(indices)],-1)
        facts=state.values.expand(len(indices),-1)
        return self.rule_gate(torch.cat([facts,structure],-1)).squeeze(-1).sigmoid()

    def fire_discrete(self,p,b,c,subset,source):
        return SoftLogic.fire(self,p,b,c,subset)*self.gates(p,source,subset)

    def infer(self,p,state,confidence=None,disabled_rule=None):
        from .logic import noisy_or,NoSupportedAction
        if not isinstance(state,DiscreteState):raise TypeError('Visual tensors cannot be passed to symbolic inference')
        state.validate();b=state.values.float();c=torch.ones_like(b);trace=[]
        # The same immutable symbol tokens condition every layer. Compute their
        # rule weights together; retain validation at the public entry boundary.
        gates=self._validated_gates(p,state,slice(0,self.action_slice.stop))
        for subset in self.layers:
            s=SoftLogic.fire(self,p,b,c,subset)*gates[subset]
            if disabled_rule is not None and subset.start<=disabled_rule<subset.stop:
                mask=torch.ones_like(s);mask[disabled_rule-subset.start]=0;s=s*mask
            contributions=s[:,None]*p.state_heads[subset].float()
            inferred=noisy_or(contributions,0)
            positive=1-(1-b*c)*(1-inferred);c=1-(1-c)*(1-inferred)
            b=(positive/c.clamp_min(1e-6)).clamp(0,1)
            trace.append({'activation':s,'contribution':contributions})
        s=SoftLogic.fire(self,p,b,c,self.action_slice)*gates[self.action_slice]
        if disabled_rule is not None and self.action_slice.start<=disabled_rule<self.action_slice.stop:
            mask=torch.ones_like(s);mask[disabled_rule-self.action_slice.start]=0;s=s*mask
        contribution=s[:,None]*p.action_heads[self.action_slice].float();support=noisy_or(contribution,0)
        if not torch.isfinite(support).all() or float(support.sum().detach())<1e-10:
            raise NoSupportedAction('No generated symbolic rule supports an action; no neural fallback')
        return {'probs':support/support.sum(),'support':support,'belief':b,'confidence':c,
                'action_activation':s,'action_contribution':contribution,'layers':trace,
                'symbol_codes':state.codes,'symbol_values':state.values,
                'rule_gates':gates[self.action_slice]}

    def transition(self,p,b,c,action,retention):
        from .logic import noisy_or
        state=tokenize(b);b=state.values;c=torch.ones_like(b)
        s=self.fire_discrete(p,b,c,self.effect_slice,state)*p.action_conditions[self.effect_slice,action].float()
        effects=noisy_or(s[:,None]*p.effects[self.effect_slice].float(),0)
        add,delete=effects[:self.k],effects[self.k:]
        old=.5+retention*(b-.5);prediction=(old+(1-old)*add)*(1-delete)
        return prediction,torch.ones_like(prediction),(add*delete).mean()

    def transition_all(self,p,b,c,retention):
        from .logic import noisy_or
        state=tokenize(b);b=state.values;c=torch.ones_like(b)
        s=self.fire_discrete(p,b,c,self.effect_slice,state)[:,None]*p.action_conditions[self.effect_slice].float()
        effects=noisy_or(s[:,:,None]*p.effects[self.effect_slice,None,:].float(),0)
        add,delete=effects[:,:self.k],effects[:,self.k:]
        old=.5+retention*(b[None]-.5);prediction=(old+(1-old)*add)*(1-delete)
        return prediction,torch.ones_like(prediction),(add*delete).mean()
