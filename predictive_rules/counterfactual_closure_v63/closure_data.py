"""Immutable, compact measured two-step outcomes from training scenes only."""
import json, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from data import sha

FIELDS = ('context','memory','observation','first_observation','first_memory',
          'second_actions','second_observation','second_memory','second_valid')


class ClosureData:
    def __init__(self, root):
        self.root=Path(root)
        read=lambda p:json.loads(p.read_text())
        self.manifest=read(self.root/'manifest.json')
        assert self.manifest['training_only'] and read(self.root/'status.json')['state']=='complete'
        self.commit=read(self.root/'source.COMMITTED.json')
        assert sha(self.root/'source.pt')==self.commit['sha256']
        groups=defaultdict(list); self.records={}
        for p in sorted((self.root/'episodes').glob('*.json')):
            r=read(p); assert r['training_only'] and r['scene'] in self.manifest['train_layouts']
            assert r['source_sha256']==self.commit['sha256']
            assert sha(p.with_suffix('.npz'))==r['sha256']
            groups[(r['domain'],r['category'],r['distance_group'])].append(r['episode'])
            self.records[r['episode']]=(p,r)
        assert set(self.records)==set(self.manifest['episode_ids']) and len(self.records)==600
        rng=random.Random(631017); held=set()
        for key,names in sorted(groups.items()):
            names=list(names); rng.shuffle(names)
            assert len(names)==50; held.update(names[:10])
        assert len(held)==120
        self.splits={part:sorted(e for e in self.records if (e in held)==(part=='holdout')) for part in ('fit','holdout')}
        arrays={part:defaultdict(list) for part in self.splits}; offsets=defaultdict(int)
        self.episode_rows={part:[] for part in self.splits}
        for part,names in self.splits.items():
            for e in names:
                path,r=self.records[e]
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
                    n=len(archive['memory']); assert n==r['positions'] and n>0
                    for k in FIELDS: arrays[part][k].append(archive[k].copy())
                self.episode_rows[part].append((offsets[part],offsets[part]+n)); offsets[part]+=n
        self.tensors={part:{k:torch.from_numpy(np.concatenate(v)).to(
            torch.float32 if k=='context' else torch.bool if k=='second_valid' else torch.long)
            for k,v in fields.items()} for part,fields in arrays.items()}
        self.fingerprint=sha(self.root/'manifest.json')

    def sample(self,rng,count,device):
        # Equal episode weight: long failures cannot dominate by their length.
        idx=[rng.randrange(*rng.choice(self.episode_rows['fit'])) for _ in range(count)]
        return self.batch('fit',torch.tensor(idx),device)

    def batch(self,part,indices,device):
        rows=self.tensors[part]
        indices=indices.to(rows['memory'].device)
        return {k:v[indices].to(device,non_blocking=True) for k,v in rows.items()}
