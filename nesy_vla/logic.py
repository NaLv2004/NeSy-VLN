"""The only action-selection path. No image encoder or trainable action head here."""
import torch


class NoSupportedAction(RuntimeError): pass


def noisy_or(x, dim):
    return -torch.expm1(torch.log1p(-x.float().clamp(0, 1-1e-6)).sum(dim))


class SoftLogic:
    def __init__(self, predicates=16, actions=17, rules=48):
        if rules % 6: raise ValueError('Rule capacity must be a multiple of six')
        self.k, self.a, self.r = predicates, actions, rules
        n = rules // 6
        self.layers = [slice(0,n), slice(n,2*n)]
        self.action_slice, self.effect_slice = slice(2*n,4*n), slice(4*n,6*n)

    def literals(self, b, confidence):
        return torch.cat([b.new_ones(1), b*confidence, (1-b)*confidence], -1)

    def fire(self, p, b, c, subset):
        evidence = torch.einsum('rjk,k->rj', p.premises[subset].float(), self.literals(b,c).float())
        return p.weights[subset].float() * evidence.prod(-1)

    def infer(self, p, belief, confidence, disabled_rule=None):
        b, c = belief.float(), confidence.float()
        trace = []
        for subset in self.layers:
            s = self.fire(p,b,c,subset)
            if disabled_rule is not None and subset.start <= disabled_rule < subset.stop:
                mask = torch.ones_like(s); mask[disabled_rule-subset.start]=0; s=s*mask
            contributions = s[:,None]*p.state_heads[subset].float()
            inferred = noisy_or(contributions, 0)
            # Two explicit layers, never reinsert this evidence in a recurrent fixed-point loop.
            positive = 1-(1-b*c)*(1-inferred)
            c = 1-(1-c)*(1-inferred)
            b = (positive / c.clamp_min(1e-6)).clamp(0,1)
            trace.append({'activation':s,'contribution':contributions})
        s = self.fire(p,b,c,self.action_slice)
        if disabled_rule is not None and self.action_slice.start <= disabled_rule < self.action_slice.stop:
            mask=torch.ones_like(s); mask[disabled_rule-self.action_slice.start]=0; s=s*mask
        contribution = s[:,None]*p.action_heads[self.action_slice].float()
        support = noisy_or(contribution,0)
        if not torch.isfinite(support).all() or float(support.sum().detach()) < 1e-10:
            raise NoSupportedAction('No rule supports an action; no fallback policy')
        probs = support/support.sum()
        return {'probs':probs,'support':support,'belief':b,'confidence':c,
                'action_activation':s,'action_contribution':contribution,'layers':trace}

    def transition(self,p,b,c,action,retention):
        s=self.fire(p,b,c,self.effect_slice)*p.action_conditions[self.effect_slice,action].float()
        effects=noisy_or(s[:,None]*p.effects[self.effect_slice].float(),0)
        add,delete=effects[:self.k],effects[self.k:]
        old=.5+retention*(b-.5)
        predicted=(old+(1-old)*add)*(1-delete)
        conf=1-(1-c*retention)*(1-torch.maximum(add,delete))
        return predicted,conf,(add*delete).mean()

    def transition_all(self,p,b,c,retention):
        """The same effect rules evaluated for all enumerated motor actions."""
        s=self.fire(p,b,c,self.effect_slice)[:,None]*p.action_conditions[self.effect_slice].float()
        effects=noisy_or(s[:,:,None]*p.effects[self.effect_slice,None,:].float(),0)
        add,delete=effects[:,:self.k],effects[:,self.k:]
        old=.5+retention*(b[None]-.5)
        predicted=(old+(1-old)*add)*(1-delete)
        conf=1-(1-c[None]*retention)*(1-torch.maximum(add,delete))
        return predicted,conf,(add*delete).mean()

    def regularization(self,p):
        effects=p.effects
        if p.effect_mode=='per_predicate':
            add,delete=effects.chunk(2,-1);effects=torch.stack([1-add-delete,add,delete],-1).clamp_min(0)
        selectors=[p.premises,p.state_heads,p.action_heads,effects,p.action_conditions]
        entropy=sum(-(v*v.clamp_min(1e-8).log()).sum(-1).mean() for v in selectors)
        return {'selector_entropy':entropy,'rule_weight':p.weights.mean()}
