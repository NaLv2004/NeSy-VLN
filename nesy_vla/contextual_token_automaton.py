"""Finite-state rules reading discrete captions through a frozen text encoder.

The cached encoder is a memoization of the LANGUAGE-ONLY model on complete
integer token sequences. It never contains multimodal/image hidden states.
"""
import torch
from torch import nn
from .probabilistic_automaton import ProbabilisticTokenAutomaton
from .semantic_token_automaton import SemanticTokens


class FrozenCaptionLookup(nn.Module):
    def __init__(self,sequences,vectors):
        super().__init__()
        keys=[tuple(s) for s in sequences]
        if len(set(keys))!=len(keys) or vectors.ndim!=2 or len(keys)!=len(vectors):
            raise ValueError('Ambiguous or malformed text-encoder cache')
        if not torch.isfinite(vectors).all():raise ValueError('Nonfinite text encoding')
        if any(not k or any(not isinstance(t,int) or t<0 for t in k) for k in keys):
            raise TypeError('Caption cache keys must be emitted integer sequences')
        self.index={k:i for i,k in enumerate(keys)}
        self.register_buffer('representations',torch.cat([vectors.detach().clone(),torch.zeros_like(vectors[:1])]),persistent=False)
        self.width=vectors.shape[-1];self.empty_index=len(keys)

    def forward(self,ids,mask):
        if ids.dtype!=torch.long or mask.dtype!=torch.bool or ids.shape!=mask.shape:
            raise TypeError('Text encoder accepts only integer tokens and a Boolean mask')
        shape=ids.shape[:-1];length=ids.shape[-1]
        rows=ids.reshape(-1,length).tolist();masks=mask.reshape(-1,length).tolist();indices=[]
        for row,active in zip(rows,masks):
            key=tuple(t for t,keep in zip(row,active) if keep)
            if not key:indices.append(self.empty_index)
            elif key not in self.index:raise KeyError('Uncached discrete caption; run the text-only encoder before inference')
            else:indices.append(self.index[key])
        return self.representations[torch.tensor(indices,device=ids.device)].float().reshape(*shape,self.width)


class ContextualTokenAutomaton(ProbabilisticTokenAutomaton):
    def __init__(self,text_encoder,**kwargs):
        super().__init__(**kwargs)
        self.text_encoder=text_encoder
        self.text_normalize=nn.LayerNorm(text_encoder.width)
        self.text_to_tokens=nn.Linear(text_encoder.width,self.patches*8)
        nn.init.zeros_(self.text_to_tokens.weight);nn.init.zeros_(self.text_to_tokens.bias)

    def initialize_common(self,state):
        target=self.state_dict()
        assert set(state)=={k for k in target if not k.startswith(('text_normalize.','text_to_tokens.'))}
        target.update(state);self.load_state_dict(target)

    def validate_codes(self,codes):
        if not isinstance(codes,SemanticTokens):raise TypeError('Expected discrete visual and caption token IDs')
        super().validate_codes(codes.visual)
        if codes.text.dtype!=torch.long or codes.mask.dtype!=torch.bool:raise TypeError('No continuous features allowed')
        if codes.text.shape!=codes.mask.shape or codes.text.shape[:-1]!=codes.visual.shape[:-1]:raise ValueError('Token dimensions disagree')
        if not 1<=codes.text.shape[-1]<=128 or (codes.text<0).any():raise ValueError('Invalid text token sequence')

    def embed(self,codes):
        self.validate_codes(codes)
        hard=self.tokens(codes.visual+self.offsets).flatten(-2)
        contextual=self.text_encoder(codes.text,codes.mask)
        delta=self.text_to_tokens(self.text_normalize(contextual))*codes.mask.any(-1,keepdim=True)
        return hard+delta
