"""Compact program synthesis from task bytes and discrete execution history.

The GRU produces a rule program, never action logits. Images in the common
generator API are deliberately unused. No task-specific predicate vocabulary
or hand-labelled automaton states are supplied.
"""
import json
import torch
from torch import nn
from .generator import ProgramDecoder


class SymbolicProgramGenerator(nn.Module):
    def __init__(self,cfg,hidden=64):
        super().__init__();self.k=cfg['k'];self.a=cfg['a']
        self.text=nn.Embedding(257,16)
        self.text_encoder=nn.GRU(16,hidden,batch_first=True)
        self.action=nn.Embedding(self.a,16)
        self.events=nn.Linear(self.k+16,hidden)
        self.history=nn.GRU(hidden,hidden,batch_first=True)
        self.head=ProgramDecoder(hidden=hidden,**cfg)

    def _events(self,feedback):
        rows=[];device=self.text.weight.device
        for event in feedback:
            codes=event.get('symbol_codes')
            if codes is None or len(codes)!=self.k or any(x not in (0,1) for x in codes):
                raise ValueError('Small program generator requires discrete predicate codes')
            action=event['action']
            if not isinstance(action,int) or not 0<=action<self.a:raise ValueError('Invalid action token')
            rows.append(torch.cat([self.text.weight.new_tensor(codes),self.action.weight[action]]))
        return torch.stack(rows)[None] if rows else None

    def context(self,task,images,feedback,previous=None):
        device=self.text.weight.device
        tokens=torch.tensor(list(task.encode('utf-8'))+[256],device=device)[None]
        text,hidden=self.text_encoder(self.text(tokens))
        events=self._events(feedback)
        if events is None:return text
        history,_=self.history(self.events(events),hidden)
        return torch.cat([text,history],1)

    def context_many(self,requests):
        return [self.context(*request) for request in requests]

    def generate_many(self,requests):
        return [self.head(h,r[3]) for h,r in zip(self.context_many(requests),requests)]

    def forward(self,task,images,feedback,previous=None,noise=None):
        return self.head(self.context(task,images,feedback,previous),previous,noise)

    def prompt(self,task,feedback,previous):
        return json.dumps({'generator':'symbolic GRU64','task':task,
            'history':[{'symbol_codes':e['symbol_codes'],'action':e['action']} for e in feedback],
            'previous_version':None if previous is None else previous.version,
            'output':'learned predicate definitions, rule premises, conclusions, effects and weights'},ensure_ascii=False)
