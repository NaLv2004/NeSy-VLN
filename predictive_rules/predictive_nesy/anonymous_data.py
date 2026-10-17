"""Task rewards are training scores, never recognized semantic predicates."""
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import torch
from .data import read,sha


def canonical_provenance(data):
    return {key:value for key,value in data.items() if key not in {'outcomes_sha256','known_risk_labels'}}


def load_windows(directory,layouts):
    directory=Path(directory);rows=read(directory/'training_rows.json');codes=np.load(directory/'visual_tokens.npy')
    manifest=read(directory/'manifest.json');interval=10;horizon=32
    if manifest['data_provenance']['train_layouts']!=sorted(layouts):raise ValueError('Training corpus layout provenance differs')
    episodes=defaultdict(list)
    for index,row in enumerate(rows):
        match=re.search(r'P25_train_(\d+)_',row.get('source_episode_id',row['episode_id']))
        if not match or 'layout_'+match.group(1) not in layouts:raise ValueError('Held-out source demonstration')
        episodes[row['episode_id']].append(index)
    windows=[]
    for episode,indices in episodes.items():
        category={rows[i]['category'] for i in indices}
        if len(category)!=1:raise ValueError('Trajectory changes its task')
        task=['chair','table'].index(next(iter(category)));actions=[int(rows[i]['label']) for i in indices]
        if actions[-1]!=0:raise ValueError('Successful source trajectory must terminate')
        for start in range(0,min(len(indices),horizon),interval):
            begin=max(0,start-interval);history_indices=indices[begin:start+1]
            history=np.zeros((interval+1,codes.shape[1]),np.int64);history[:len(history_indices)]=codes[history_indices]
            previous=np.full(interval+1,17,np.int64);previous[:len(history_indices)]=[actions[t-1] if t else 17 for t in range(begin,start+1)]
            future_indices=indices[start:start+interval+1];n=len(future_indices)
            future=np.zeros_like(history);future[:n]=codes[future_indices];future[n:]=future[n-1]
            chosen=np.zeros(interval+1,np.int64);chosen[:n]=actions[start:start+n]
            valid=np.arange(interval+1)<n;supervised=np.zeros(interval+1,bool)
            supervised[:n]=[rows[i].get('supervised',True) for i in future_indices]
            remaining=np.maximum(len(indices)-1-start-np.arange(interval+1),0)
            discount=(.97**remaining).astype(np.float32)
            scores=np.full(interval+1,-.02,np.float32);scores[chosen==0]=1.
            returns=(discount-.02*(1-discount)/(1-.97)).astype(np.float32)
            windows.append(dict(episode_id=episode,task=task,start=start,history=history,previous_actions=previous,
                history_length=len(history_indices),targets=future,actions=chosen,valid=valid,supervised=supervised,
                scalar_rewards=scores,returns=returns))
    provenance=dict(source=str(directory),rows=len(rows),episodes=len(episodes),windows=len(windows),
        rows_sha256=sha(directory/'training_rows.json'),codes_sha256=sha(directory/'visual_tokens.npy'),
        source_manifest_sha256=sha(directory/'manifest.json'),train_layouts=sorted(layouts),interval=interval,horizon=horizon,
        future_frames='supervision_only',boundary_context='current_and_past_only')
    return windows,provenance


def batch(windows,device):
    result={'episode_id':[w['episode_id'] for w in windows]}
    for key in ['task','start','history','previous_actions','history_length','targets','actions','valid','supervised',
                'teacher_ids','scalar_rewards','returns']:
        result[key]=torch.as_tensor(np.asarray([w[key] for w in windows]),device=device)
    return result


class OutcomeBank:
    def __init__(self,directory,layouts):
        self.directory=Path(directory);self.manifest=read(self.directory/'manifest.json')
        if not self.manifest['training_only'] or set(layouts)!=set(self.manifest['data']['train_layouts']):raise ValueError('Training-only split required')
        self.layouts=set(layouts);self.allowed=set(self.manifest['episodes']);self.episodes={};self.keys=[];self.by_key={}
        self.refresh()

    def refresh(self):
        for path in sorted(self.directory.glob('*.npz')):
            if path.name.endswith('.tmp.npz') or path.stem in self.episodes:continue
            match=re.search(r'P25_train_(\d+)_',path.stem)
            if path.stem not in self.allowed or match is None or 'layout_'+match.group(1) not in self.layouts:raise ValueError('Held-out example')
            # Only images, actions and scalar task scores cross this adapter.
            # Legacy metadata is not returned to either the model or objective.
            with np.load(path,allow_pickle=False) as source:
                d={key:source[key].copy() for key in ['codes','successors','actions','probe_valid','task']}
                n=len(d['actions'])
                if self.manifest.get('schema')=='anonymous_scalar_outcomes_v1':
                    reward=source['scalar_rewards'].copy()
                    if reward.shape!=(n,17) or not np.isfinite(reward).all():raise ValueError('Invalid scalar task scores')
                else:
                    reward=np.full((n,17),-.02,np.float32)
                    reward[:,0]=np.where(source['eligible_current'],1.,-1.)
                d['scalar_rewards']=reward
            self.episodes[path.stem]=d
            for t in range(n):
                if not d['probe_valid'][t]:continue
                key=(t//10*10,t%10);self.keys.append(key);self.by_key.setdefault(key,[]).append((path.stem,t))
        return dict(episodes=len(self.episodes),states=len(self.keys),manifest_sha256=sha(self.directory/'manifest.json'))

    def sample(self,rng,size,device):
        start,offset=rng.choice(self.keys);chosen=rng.choices(self.by_key[start,offset],k=size);rows=[]
        for name,t in chosen:
            d=self.episodes[name];begin=max(0,start-10)
            rows.append(dict(task=int(d['task']),history=d['codes'][begin:start+1],
                previous_actions=np.array([int(d['actions'][j-1]) if j else 17 for j in range(begin,start+1)]),
                prefix_codes=d['codes'][start:t+1],prefix_actions=d['actions'][start:t],
                successors=d['successors'][t],scalar_rewards=d['scalar_rewards'][t]))
        result={key:torch.as_tensor(np.asarray([r[key] for r in rows]),device=device) for key in rows[0]}
        for key in ['task','history','previous_actions','prefix_codes','prefix_actions','successors']:result[key]=result[key].long()
        result.update(start=start,offset=offset,episode_ids=[n for n,t in chosen]);return result
