"""Total generated decision rules over discrete learned facts.

Every internal node asks a learned predicate query. Its two complementary
branches form exhaustive cases. A leaf is a conjunction of those branch
conditions and concludes the generated action distribution. No RGB input or
fallback action head exists. Predicate meanings and node assignments are learned.
"""
import math
import torch
from .neural_logic import NeuralSymbolicLogic
from .symbols import DiscreteState
from .logic import NoSupportedAction


class TotalBranchLogic(NeuralSymbolicLogic):
    def __init__(self,predicates=32,actions=17,rules=96):
        super().__init__(predicates,actions,rules)
        leaves=self.action_slice.stop-self.action_slice.start
        if leaves&(leaves-1):raise ValueError('Power-of-two leaf count required')
        self.depth=int(math.log2(leaves));self.node_count=leaves-1

    def probabilities_for_states(self,p,states):
        """Vectorized execution of the same proof on discrete counterfactuals.

        This is a training diagnostic/estimator, not a different action policy.
        No continuous observation tensor is accepted at this boundary.
        """
        if not isinstance(states,DiscreteState):raise TypeError('Expected discrete predicate states')
        states.validate()
        if states.values.ndim!=2 or states.values.shape[1]!=self.k:raise ValueError('Expected N by K states')
        if p.action_semantics!='branch_tree':raise ValueError('Expected generated branch rules')
        b=states.values.float();literals=torch.cat([b.new_ones((len(b),1)),b,1-b],-1)
        query=(literals@p.premises[:self.node_count,0].float().T).clamp(0,1)
        mass=b.new_ones((len(b),1))
        for depth in range(self.depth):
            start=2**depth-1;tests=query[:,start:start+2**depth]
            mass=torch.stack([mass*tests,mass*(1-tests)],-1).flatten(1)
        support=mass@p.action_heads[self.action_slice].float();total=support.sum(-1,keepdim=True)
        if not torch.isfinite(support).all() or bool((total<=0).any()):
            raise NoSupportedAction('No generated branch supports a counterfactual state')
        return support/total

    def infer(self,p,state,confidence=None,disabled_rule=None):
        if not isinstance(state,DiscreteState):raise TypeError('Branch rules accept discrete predicate states only')
        state.validate()
        if p.action_semantics!='branch_tree':raise ValueError('Program is not a branch-tree program')
        b=state.values.float();c=torch.ones_like(b)
        query=p.premises[:self.node_count,0].float()@self.literals(b,c)
        if not torch.isfinite(query).all() or bool(((query< -1e-5)|(query>1+1e-5)).any()):
            raise ValueError('Invalid generated predicate selector')
        query=query.clamp(0,1);mass=b.new_ones(1);layers=[]
        for depth in range(self.depth):
            start=2**depth-1;tests=query[start:start+2**depth]
            mass=torch.stack([mass*tests,mass*(1-tests)],-1).flatten()
            layers.append({'activation':mass,'node_queries':tests})
        if disabled_rule is not None and self.action_slice.start<=disabled_rule<self.action_slice.stop:
            mask=torch.ones_like(mass);mask[disabled_rule-self.action_slice.start]=0;mass=mass*mask
        contribution=mass[:,None]*p.action_heads[self.action_slice].float()
        support=contribution.sum(0);total=support.sum()
        if not torch.isfinite(support).all() or float(total.detach())<=0:
            raise NoSupportedAction('No surviving branch rule; no fallback')
        return {'probs':support/total,'support':support,'belief':b,'confidence':c,
                'action_activation':mass,'action_contribution':contribution,'layers':layers,
                'symbol_codes':state.codes,'symbol_values':state.values,'rule_gates':torch.ones_like(mass),
                'branch_queries':query,'coverage_mass':total}
