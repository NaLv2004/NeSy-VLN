"""Successful demonstrations plus train-only on-policy world-model probes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import numpy as np
import torch

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')

def read(path):return json.loads(Path(path).read_text())

def sha(path):
    result=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):result.update(chunk)
    return result.hexdigest()

class Corpus:
    def __init__(self, rank=0,world_size=1,branch_root=None):
        self.manifest=read(ROOT/'run_action_grounded_v33/manifest.json')
        self.layouts=set(self.manifest['data']['train_layouts'])
        source=Path(self.manifest['data']['source'])
        assert sha(source/'training_rows.json')==self.manifest['data']['rows_sha256']
        self.rows=read(source/'training_rows.json')
        self.raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
        cache=read(ROOT/'prequant_visual_features_v18/manifest.json')
        assert self.raw.shape==(len(self.rows),49,128)
        self.episodes=defaultdict(list)
        for i,row in enumerate(self.rows):
            self.check_layout(row.get('source_episode_id',row['episode_id']))
            self.episodes[row['episode_id']].append(i)
        self.names=sorted(self.episodes)
        for name,indices in self.episodes.items():
            assert self.rows[indices[-1]]['label']==0 and len(indices)<=33
        self.branches=[];self.branch_positions=[]
        metas=sorted((ROOT/'raw_successor_features_v31').glob('counterfactual_*/*.json'))
        for meta in metas[rank::world_size]:
            record=read(meta);self.check_layout(record['episode'])
            path=meta.with_suffix('.npz')
            if not path.exists():continue
            assert sha(path)==record['output_sha256']
            with np.load(path,allow_pickle=False) as archive:
                d={k:archive[k] for k in ('features','successor_features','recorded_actions','probe_valid')}
            idx=len(self.branches);self.branches.append((record,d))
            self.branch_positions.extend((idx,t) for t,v in enumerate(d['probe_valid']) if v and t<32)
        assert self.branch_positions
        self.old_branch_positions=list(self.branch_positions)
        self.onpolicy_positions=[]
        extra_provenance=None
        if branch_root:
            extra=Path(branch_root).resolve()
            assert extra.parent==ROOT and extra.name=='onpolicy_successors_v40'
            extra_provenance=read(extra/'manifest.json')
            assert extra_provenance['training_only'] and extra_provenance['no_goal_or_imitation_supervision']
            assert set(extra_provenance['train_layouts'])==self.layouts
            assert read(extra/'status.json')['state']=='complete','Pin a complete training cache before using it'
            metas=sorted((extra/'counterfactual_onpolicy').glob('*.json'))
            assert len(metas)==len(extra_provenance['episode_ids'])
            assert {read(p)['episode'] for p in metas}==set(extra_provenance['episode_ids'])
            for meta in metas[rank::world_size]:
                record=read(meta);self.check_layout(record['episode'])
                assert record['training_only'] and not record['goal_supervision']
                assert record['scene'] in self.layouts and not record['probe_labels_available_to_policy']
                path=meta.with_suffix('.npz');assert sha(path)==record['output_sha256']
                assert record['source_checkpoint_sha256']==extra_provenance['source_commit']['sha256']
                with np.load(path,allow_pickle=False) as archive:
                    d={k:archive[k] for k in ('features','successor_features','recorded_actions','probe_valid')}
                n=len(d['features']);assert 1<=n<=32 and d['features'].shape==(n,49,128)
                assert d['successor_features'].shape==(n,17,49,128)
                assert len(d['recorded_actions']) in (n,n-1)
                assert np.isfinite(d['features']).all() and np.isfinite(d['successor_features']).all()
                idx=len(self.branches);self.branches.append((record,d))
                self.onpolicy_positions.extend((idx,t) for t,v in enumerate(d['probe_valid']) if v and t<32)
            assert self.onpolicy_positions
            self.branch_positions.extend(self.onpolicy_positions)
        self.effect_evidence=read(ROOT/'src_discrete_closure_v38/effect_invariants_audit.json')
        assert self.effect_evidence['training_only'] and len(self.effect_evidence['fit_changed'])==17
        # Only the fitting partition induces the class. The held-out partition
        # checks it and supplies no optimization labels.
        self.operator_targets=np.asarray([count==0 for count in self.effect_evidence['fit_changed']],np.int64)
        self.provenance=dict(demonstrations=len(self.names),frames=len(self.rows),
            branch_episodes_local=len(self.branches),branch_positions_local=len(self.branch_positions),
            raw_cache=str(ROOT/'raw_successor_features_v31'),feature_cache=str(ROOT/'prequant_visual_features_v18'),
            train_layouts=sorted(self.layouts),data=self.manifest['data'],feature_manifest=cache,
            operator_class_evidence=self.effect_evidence,onpolicy_cache=extra_provenance,
            onpolicy_positions_local=len(self.onpolicy_positions),branch_onpolicy_fraction=.5 if branch_root else 0.)

    def check_layout(self,name):
        match=re.search(r'P25_train_(\d+)_',name)
        assert match and 'layout_'+match.group(1) in self.layouts, name

    def sample(self,rng,size,branch_size,device):
        names=rng.choices(self.names,k=size)
        length=34
        features=np.zeros((size,length,49,128),np.float16)
        previous=np.full((size,length),17,np.int64)
        actions=np.zeros((size,length-1),np.int64)
        valid=np.zeros((size,length-1),bool)
        starts=[];ends=[];tasks=[]
        for b,name in enumerate(names):
            indices=self.episodes[name];n=len(indices)
            features[b,:n]=self.raw[indices];features[b,n]=features[b,n-1]
            labels=[int(self.rows[i]['label']) for i in indices]
            actions[b,:n]=labels;previous[b,1:n+1]=labels
            valid[b,:n]=[self.rows[i].get('supervised',True) for i in indices]
            starts.append(rng.randrange(n));ends.append(n)
            tasks.append(['chair','table'].index(self.rows[indices[0]]['category']))
        result=dict(features=features,previous=previous,actions=actions,valid=valid,
            start=np.asarray(starts),end=np.asarray(ends),task=np.asarray(tasks))
        bfeatures=np.zeros((branch_size,33,49,128),np.float16)
        bprevious=np.full((branch_size,33),17,np.int64)
        successors=np.zeros((branch_size,17,49,128),np.float16)
        bt=[];bs=[];tasks=[]
        if self.onpolicy_positions:
            count=branch_size//2
            chosen=[(idx,t,True) for idx,t in rng.choices(self.onpolicy_positions,k=count)]
            chosen += [(idx,t,False) for idx,t in rng.choices(self.old_branch_positions,k=branch_size-count)]
            rng.shuffle(chosen)
        else:
            chosen=[(idx,t,False) for idx,t in rng.choices(self.old_branch_positions,k=branch_size)]
        onpolicy=[]
        for b,(idx,t,is_onpolicy) in enumerate(chosen):
            record,d=self.branches[idx]
            bfeatures[b,:t+1]=d['features'][:t+1]
            bprevious[b,1:t+1]=d['recorded_actions'][:t]
            successors[b]=d['successor_features'][t]
            bt.append(t);bs.append(rng.randrange(t+1))
            tasks.append(0 if '_chair_' in record['episode'] else 1)
            onpolicy.append(is_onpolicy)
        result.update(branch_features=bfeatures,branch_previous=bprevious,successors=successors,
            branch_t=np.asarray(bt),branch_start=np.asarray(bs),branch_task=np.asarray(tasks),branch_onpolicy=np.asarray(onpolicy))
        result['operator_targets']=self.operator_targets
        return {k:torch.as_tensor(v,device=device) for k,v in result.items()}
