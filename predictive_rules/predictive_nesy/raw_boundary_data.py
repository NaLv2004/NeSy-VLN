"""Exact causal image correspondence for training-only boundary features."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .data import sha

class RawBoundaryFeatures:
    def __init__(self,cache,device):
        cache=Path(cache);self.manifest=json.loads((cache/'manifest.json').read_text())
        source=Path(self.manifest['source']);assert sha(cache/'features.npy')==self.manifest['feature_sha256']
        assert sha(source/'training_rows.json')==self.manifest['row_manifest_sha256']
        assert sha(source/'visual_tokens.npy')==self.manifest['source_token_sha256']
        self.features=torch.tensor(np.load(cache/'features.npy'),device=device)
        self.tokens=torch.tensor(np.load(source/'visual_tokens.npy'),device=device).long();self.groups=defaultdict(list)
        for index,row in enumerate(json.loads((source/'training_rows.json').read_text())):self.groups[row['episode_id']].append(index)

    def indices(self,episode,start):
        group=self.groups.get(episode)
        if group is None or start>=len(group):return None
        return group[max(0,start-10):start+1]

    def demonstrations(self,windows):
        index=[self.indices(w['episode_id'],w['start']) for w in windows]
        if any(ix is None for ix in index):raise ValueError('Demonstration image correspondence missing')
        ix=torch.tensor(index,device=self.features.device)
        expected=torch.tensor(np.asarray([w['history'][:len(index[0])] for w in windows]),device=ix.device)
        if not torch.equal(self.tokens[ix],expected):raise ValueError('Wrong causal demonstration image sequence')
        return self.features[ix],torch.ones(len(windows),device=ix.device,dtype=torch.bool)

    def transitions(self,data):
        b,width=data['history'].shape[:2];device=self.features.device
        indices=np.zeros((b,width),np.int64);known=np.zeros(b,bool)
        for i,name in enumerate(data['episode_ids']):
            ix=self.indices(name,data['start'])
            # Later own-policy histories are not silently paired to a different
            # demonstration. Missing exact correspondence keeps the old input.
            if ix is None or len(ix)!=width:continue
            indices[i]=ix;known[i]=True
        ids=torch.tensor(indices,device=device)
        available=torch.tensor(known,device=device)&(self.tokens[ids]==data['history']).all(-1).all(-1)
        features=torch.where(available[:,None,None,None],self.features[ids],0.)
        return features,available
