"""Aligned raw demonstration and counterfactual features on train layouts."""
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np
import torch
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures
from predictive_nesy.data import sha


class RawFeatures(RawBoundaryFeatures):
    def targets(self,rows):
        indices=[]
        for row in rows:
            group=self.groups[row['episode_id']];start=row['start']
            indices.append([group[min(start+t,len(group)-1)] for t in range(11)])
        ids=torch.tensor(indices,device=self.features.device)
        expected=torch.tensor(np.asarray([r['targets'] for r in rows]),device=ids.device)
        assert torch.equal(self.tokens[ids],expected)
        return self.features[ids]


class RawBranchBank:
    def __init__(self,root,layouts):
        self.root=Path(root);self.layouts=set(layouts);self.loaded={};self.groups=defaultdict(list);self.keys=[];self.refresh()

    def refresh(self):
        for meta in sorted(self.root.glob('counterfactual_*/*.json')):
            key=str(meta)
            if key in self.loaded:continue
            record=json.loads(meta.read_text());path=meta.with_suffix('.npz')
            match=re.search(r'P25_train_(\d+)_',record['episode'])
            assert match and 'layout_'+match.group(1) in self.layouts
            if not path.exists():continue
            assert sha(path)==record['output_sha256']
            with np.load(path,allow_pickle=False) as source:d={name:source[name].copy() for name in source.files}
            assert d['features'].shape==(len(d['recorded_actions']),49,128)
            self.loaded[key]=(record,d)
            for t,valid in enumerate(d['probe_valid']):
                if valid:
                    self.groups[t].append((key,t));self.keys.append(t)
        return dict(episodes=len(self.loaded),positions=len(self.keys))

    def sample(self,rng,size,device,episode_lookup):
        # A runtime program may begin at any observation, so all-action
        # training must not assume the obsolete fixed ten-step boundaries.
        # The complete recorded prefix lets us choose a causal boundary without
        # rendering new data or changing the successor images.
        time_index=rng.choice(self.keys)
        start=rng.randint(0,time_index);offset=time_index-start
        selected=rng.choices(self.groups[time_index],k=size);rows=[]
        for key,t in selected:
            record,d=self.loaded[key];begin=max(0,start-10)
            rows.append(dict(task=['chair','table'].index(episode_lookup[record['episode']]['category']),
                history=d['recorded_codes'][begin:start+1],previous_actions=np.asarray([int(d['recorded_actions'][j-1]) if j else 17 for j in range(begin,start+1)]),
                boundary_visual=d['features'][begin:start+1],prefix_codes=d['recorded_codes'][start:t+1],prefix_visual=d['features'][start:t+1],
                successors=d['recorded_successors'][t],successor_visual=d['successor_features'][t]))
        data={key:torch.as_tensor(np.asarray([r[key] for r in rows]),device=device) for key in rows[0]}
        for key in ['task','history','previous_actions','prefix_codes','successors']:data[key]=data[key].long()
        data.update(start=start,offset=offset)
        return data
