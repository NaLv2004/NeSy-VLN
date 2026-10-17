"""Canonical raw features from each transition bank's actual action history."""
import json
from pathlib import Path
import numpy as np
import torch
from .raw_boundary_data import RawBoundaryFeatures
from .data import sha

class ReplayedBoundaryFeatures(RawBoundaryFeatures):
    def __init__(self, cache, replay, device):
        super().__init__(cache, device)
        replay=Path(replay); self.replay_manifest=json.loads((replay/'manifest.json').read_text())
        assert self.replay_manifest['training_only']
        self.lookup={}; arrays=[]; tokens=[]; available=[]; offset=0
        for record in self.replay_manifest['records']:
            path=replay/record['bank']/(record['episode']+'.npz')
            assert sha(path)==record['output_sha256']
            original=replay.parent/record['bank']/(record['episode']+'.npz')
            assert sha(original)==record['source_sha256']
            with np.load(path,allow_pickle=False) as data:
                raw=data['features'].copy(); code=data['recorded_codes'].copy(); avail=data['context_available'].copy()
                with np.load(original,allow_pickle=False) as source:
                    assert np.array_equal(code,source['codes']) and np.array_equal(data['recorded_actions'],source['actions'])
                for start in range(0,len(code),10):
                    if avail[start]:assert np.array_equal(data['replayed_codes'][max(0,start-10):start+1],code[max(0,start-10):start+1])
            key=(record['bank'],record['episode']);assert key not in self.lookup
            self.lookup[key]=(offset,len(code));offset+=len(code)
            arrays.append(raw);tokens.append(code);available.append(avail)
        self.replayed=torch.tensor(np.concatenate(arrays),device=device)
        self.replayed_tokens=torch.tensor(np.concatenate(tokens),device=device).long()
        self.replayed_available=torch.tensor(np.concatenate(available),device=device)

    def transitions(self,data,bank):
        width=data['history'].shape[1];start=data['start'];device=self.replayed.device
        indices=[];current=[]
        for episode in data['episode_ids']:
            offset,length=self.lookup[(Path(bank).name,episode)]
            assert start<length and width==min(start+1,11)
            indices.append(list(range(offset+max(0,start-10),offset+start+1)));current.append(offset+start)
        ids=torch.tensor(indices,device=device);last=torch.tensor(current,device=device)
        assert torch.equal(self.replayed_tokens[ids],data['history']), 'Replay bank/history mismatch'
        availability=self.replayed_available[last]
        return torch.where(availability[:,None,None,None],self.replayed[ids],0.),availability
