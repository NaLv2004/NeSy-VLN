from dataclasses import dataclass, fields
import torch


@dataclass
class Program:
    definitions: torch.Tensor
    premises: torch.Tensor
    state_heads: torch.Tensor
    action_heads: torch.Tensor
    effects: torch.Tensor
    action_conditions: torch.Tensor
    weights: torch.Tensor
    version: int = 0
    effect_mode: str = 'categorical'
    action_semantics: str = 'horn'
    conjunctive_conditions: torch.Tensor | None = None
    conjunction_operator: str = 'product'
    neural_conjunction_gates: bool = False

    def map(self, fn):
        return Program(**{f.name: fn(getattr(self, f.name)) if torch.is_tensor(getattr(self,f.name))
                          else getattr(self,f.name) for f in fields(self)})

    def detach(self):
        return self.map(lambda x: x.detach())

    def cpu(self):
        return self.map(lambda x: x.detach().float().cpu())

    def to(self, device):
        return self.map(lambda x: x.to(device))

    def record(self):
        return {f.name: getattr(self, f.name).detach().float().cpu().tolist()
                if torch.is_tensor(getattr(self,f.name)) else getattr(self,f.name) for f in fields(self)
                if getattr(self,f.name) is not None}

    @classmethod
    def from_record(cls,record,device='cpu'):
        return cls(**{k:torch.tensor(v,device=device) if k not in ('version','effect_mode','action_semantics','conjunction_operator','neural_conjunction_gates') else v
                      for k,v in record.items()})

    def validate(self):
        for f in fields(self):
            if f.name in ('version','effect_mode','action_semantics','conjunction_operator','neural_conjunction_gates') or getattr(self,f.name) is None: continue
            x = getattr(self, f.name)
            if not torch.isfinite(x).all(): raise FloatingPointError(f.name)
        if self.effect_mode not in ('categorical','per_predicate'):raise ValueError('Unknown effect mode')
        if self.conjunction_operator not in ('product','minimum'):raise ValueError('Unknown conjunction operator')
        if type(self.neural_conjunction_gates) is not bool:raise ValueError('Neural conjunction gate flag must be Boolean')
        if self.neural_conjunction_gates and self.action_semantics!='conjunctive_rules':raise ValueError('Gates require conjunction rules')
        if self.action_semantics not in ('horn','branch_tree','conjunctive_rules'):raise ValueError('Unknown action semantics')
        if self.action_semantics=='conjunctive_rules':
            x=self.conjunctive_conditions
            if x is None or x.shape!=(len(self.weights),len(self.definitions),3):raise ValueError('Missing conjunction conditions')
            if (x<0).any() or not torch.allclose(x.sum(-1),torch.ones_like(x.sum(-1)),atol=1e-4):raise ValueError('Invalid conjunction conditions')
        for name in ['premises', 'state_heads', 'action_heads', 'action_conditions']+(['effects'] if self.effect_mode=='categorical' else []):
            x = getattr(self, name)
            if (x < 0).any() or not torch.allclose(x.sum(-1), torch.ones_like(x.sum(-1)), atol=1e-4):
                raise ValueError('Invalid selector: ' + name)
        if self.effect_mode=='per_predicate':
            add,delete=self.effects.chunk(2,-1)
            if (self.effects<0).any() or (add+delete>1.0001).any():raise ValueError('Invalid per-predicate effects')

    def harden(self):
        import torch.nn.functional as F
        result=self.map(lambda x:x)
        for name in ['premises','state_heads','action_heads','action_conditions']:
            x=getattr(self,name);setattr(result,name,F.one_hot(x.argmax(-1),x.shape[-1]).float())
        if self.effect_mode=='categorical':result.effects=F.one_hot(self.effects.argmax(-1),self.effects.shape[-1]).float()
        else:
            add,delete=self.effects.chunk(2,-1)
            choices=torch.stack([1-add-delete,add,delete],-1).argmax(-1)
            result.effects=torch.cat([(choices==1).float(),(choices==2).float()],-1)
        if self.conjunctive_conditions is not None:
            result.conjunctive_conditions=F.one_hot(self.conjunctive_conditions.argmax(-1),3).float()
        return result


def describe(program, limit=48):
    p = program.cpu(); rows = []
    if p.action_semantics=='conjunctive_rules':
        n=len(p.weights)//6
        return {'version':p.version,'action_semantics':p.action_semantics,'effect_mode':p.effect_mode,'conjunction_operator':p.conjunction_operator,
            'neural_conjunction_gates':p.neural_conjunction_gates,
            'literal_convention':'0=true; 1..K=positive; K+1..2K=negative',
            'condition_order':['unused','required_true','required_false'],
            'action_rules':[{'id':i,'mandatory_literal_mixture':p.premises[i,0].tolist(),
                'predicate_conditions':p.conjunctive_conditions[i].tolist(),
                'weight':float(p.weights[i]),'action_distribution':p.action_heads[i].tolist()} for i in range(2*n,4*n)],
            'execution':'Combine mandatory and generated predicate-condition matches. Optional learned gates consult hard symbols and rule structure only; false premises remain unsupported. Normalize supported rule masses, then combine generated action conclusions.'}
    if p.action_semantics=='branch_tree':
        n=p.weights.numel()//6;leaves=2*n;nodes=leaves-1
        branches=[{'id':i,'literal_indices':p.premises[i,0].nonzero().flatten().tolist(),
                   'literal_mixture':p.premises[i,0].tolist(),'true_child':2*i+1,'false_child':2*i+2}
                  for i in range(nodes)]
        paths=[]
        for leaf in range(leaves):
            cursor=leaf+nodes;path=[]
            while cursor:
                parent=(cursor-1)//2;path.append({'node':parent,'true_branch':cursor==2*parent+1});cursor=parent
            paths.append({'id':2*n+leaf,'conditions':list(reversed(path)),
                          'action_distribution':p.action_heads[2*n+leaf].tolist()})
        return {'version':p.version,'action_semantics':'branch_tree','effect_mode':p.effect_mode,
                'literal_convention':'0=true; 1..K=positive; K+1..2K=negative',
                'nodes':branches,'action_rules':paths,
                'coverage':'Each learned predicate test has complementary true/false successors; leaf path masses sum to one.'}
    for r in range(min(limit, p.weights.numel())):
        rows.append({'id': r, 'weight': round(float(p.weights[r]), 4),
            'premise_indices': p.premises[r].argmax(-1).tolist(),
            'premise_peaks': p.premises[r].max(-1).values.tolist(),
            'state_head': int(p.state_heads[r].argmax()),
            'action_head': int(p.action_heads[r].argmax()),
            'effect_head': int(p.effects[r].argmax()),
            'action_condition': int(p.action_conditions[r].argmax())})
        if p.effect_mode=='per_predicate':
            k=p.definitions.shape[0];hard=p.harden().effects[r]
            rows[-1]['add_predicates']=torch.where(hard[:k]>0)[0].tolist()
            rows[-1]['delete_predicates']=torch.where(hard[k:]>0)[0].tolist()
            rows[-1].pop('effect_head')
    return {'version': p.version,'effect_mode':p.effect_mode, 'literal_convention': '0=true; 1..K=positive; K+1..2K=negative',
            'rules': rows}
