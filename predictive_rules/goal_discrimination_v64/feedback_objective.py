"""Supervise existing anonymous goal-code equality; no added policy head."""
import json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import hard
from objectives import objective
from data import sha

class FeedbackData:
    def __init__(self,root,holdout,device):
        self.root=Path(root);self.device=device;self.holdout=set(holdout)
        assert json.loads((self.root/'status.json').read_text())['state']=='complete'
        self.names=[];xs=[];prev=[];tasks=[];self.parts={'fit':[],'holdout':[]}
        self.pools={part:{False:[],True:[]} for part in self.parts}
        for meta in sorted((self.root/'episodes').glob('*.json')):
            r=json.loads(meta.read_text());p=meta.with_suffix('.npz')
            assert r['training_only'] and sha(p)==r['sha256']
            with np.load(p,allow_pickle=False) as d:
                x=np.zeros((34,49,128),np.float16);x[:len(d['features'])]=d['features']
                actions=d['actions'].copy();y=d['stop_success'].copy();boundaries=d['boundaries'].copy()
            previous=np.full(34,17,np.int64);previous[1:len(actions)+1]=actions
            n=len(actions)+(not len(actions) or actions[-1]!=0)
            part='holdout' if r['episode'] in self.holdout else 'fit';i=len(xs)
            for t in range(n):
                boundary=int(boundaries[boundaries<=t][-1]);row=(i,t,boundary,bool(y[t]))
                self.parts[part].append(row);self.pools[part][bool(y[t])].append(row)
            xs.append(x);prev.append(previous);tasks.append(['chair','table'].index(r['category']));self.names.append(r['episode'])
        assert len(self.names)==600 and len(self.holdout)==120
        assert all(self.pools[part][v] for part in self.parts for v in (False,True))
        self.features=torch.from_numpy(np.stack(xs)).to(device)
        self.previous=torch.tensor(np.stack(prev),device=device)
        self.tasks=torch.tensor(tasks,device=device)
    def batch(self,rows):
        index=torch.tensor([r[0] for r in rows],device=self.device)
        return dict(features=self.features[index],previous=self.previous[index],task=self.tasks[index],
            current=torch.tensor([r[1] for r in rows],device=self.device),
            boundary=torch.tensor([r[2] for r in rows],device=self.device),
            success=torch.tensor([r[3] for r in rows],device=self.device))
    def sample(self,rng,n):
        rows=[rng.choice(self.pools['fit'][bool(i%2)]) for i in range(n)];rng.shuffle(rows)
        return self.batch(rows)


def feedback(model,d):
    x=d['features'];b=torch.arange(len(x),device=x.device)
    global_o=hard(model.observe(x))
    context=model.context_sequence(x,global_o,d['previous'],d['task'])[b,d['boundary']]
    logits=model.observe(x[b,d['current']],context)
    _,goal=model.goal_logits(context,d['task'])
    # The label names no state ID. The model's own generated goal chooses the
    # anonymous target code. Existing demonstration loss trains that generator.
    code=goal.detach().argmax(-1)
    logp=logits.log_softmax(-1).gather(-1,code[:,None]).squeeze(-1)
    p=logp.exp().clamp(max=1-1e-6);y=d['success']
    loss=torch.where(y,-logp,-torch.log1p(-p)).mean()
    pred=logits.argmax(-1)==code
    metrics=dict(feedback_loss=loss.detach(),positive_count=y.sum(),negative_count=(~y).sum(),
        true_positive=(pred&y).sum(),false_positive=(pred&~y).sum(),
        positive_nll=(-logp*y).sum(),negative_nll=(-torch.log1p(-p)*(~y)).sum(),
        feedback_observation_codes=logits.argmax(-1).unique().numel())
    return loss,{k:torch.as_tensor(v,device=x.device).detach() for k,v in metrics.items()}


class FeedbackObjective(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,base,measured):
        loss,metrics=objective(self.model,base)
        extra,more=feedback(self.model,measured)
        total=loss+extra
        metrics.update(more);metrics['base_loss']=loss.detach();metrics['loss']=total.detach()
        return total,metrics
