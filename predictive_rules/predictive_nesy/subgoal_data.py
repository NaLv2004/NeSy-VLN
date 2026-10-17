"""Anonymous future-state demonstrations and visual transition examples."""
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import torch
from .data import read,sha


def canonical_provenance(data):
    return {key:value for key,value in data.items() if key not in {'outcomes_sha256','known_risk_labels'}}


FIELDS={'task','start','history','previous_actions','history_length','targets','actions','action_valid','state_valid',
    'supervised','goal_index','episode_id','offset','prefix_codes','prefix_actions','successors','episode_ids'}


def require_state_only(batch):
    if set(batch)-FIELDS:raise ValueError('Unexpected training fields: '+str(sorted(set(batch)-FIELDS)))


def load_windows(directory,layouts):
    directory=Path(directory);rows=read(directory/'training_rows.json');codes=np.load(directory/'visual_tokens.npy')
    manifest=read(directory/'manifest.json');episodes=defaultdict(list);result=[]
    if manifest['data_provenance']['train_layouts']!=sorted(layouts):raise ValueError('Training corpus layout provenance differs')
    for index,row in enumerate(rows):
        match=re.search(r'P25_train_(\d+)_',row.get('source_episode_id',row['episode_id']))
        if not match or 'layout_'+match.group(1) not in layouts:raise ValueError('Held-out source demonstration')
        episodes[row['episode_id']].append(index)
    for episode,indices in episodes.items():
        category={rows[i]['category'] for i in indices}
        if len(category)!=1:raise ValueError('Trajectory changes its task')
        task=['chair','table'].index(next(iter(category)));actions=[int(rows[i]['label']) for i in indices]
        if actions[-1]!=0:raise ValueError('Successful source trajectory must terminate')
        for start in range(0,min(len(indices),32),10):
            begin=max(0,start-10);history_indices=indices[begin:start+1]
            history=np.zeros((11,codes.shape[1]),np.int64);history[:len(history_indices)]=codes[history_indices]
            previous=np.full(11,17,np.int64);previous[:len(history_indices)]=[actions[t-1] if t else 17 for t in range(begin,start+1)]
            h=min(len(indices)-start,10,32-start)
            # The RGB following STOP equals the final input RGB. Its anonymous
            # memory code is learned using the executed action; no ID is reserved.
            future=codes[[indices[min(start+i,len(indices)-1)] for i in range(11)]].astype(np.int64)
            chosen=np.zeros(10,np.int64);chosen[:h]=actions[start:start+h]
            supervised=np.zeros(10,bool);supervised[:h]=[rows[indices[start+i]].get('supervised',True) for i in range(h)]
            result.append(dict(episode_id=episode,task=task,start=start,history=history,previous_actions=previous,
                history_length=len(history_indices),targets=future,actions=chosen,action_valid=np.arange(10)<h,
                state_valid=np.arange(11)<=h,supervised=supervised,goal_index=h))
    provenance=dict(source=str(directory),rows=len(rows),episodes=len(episodes),windows=len(result),
        rows_sha256=sha(directory/'training_rows.json'),codes_sha256=sha(directory/'visual_tokens.npy'),
        source_manifest_sha256=sha(directory/'manifest.json'),train_layouts=sorted(layouts),interval=10,horizon=32,
        future_frames='supervision_only',boundary_context='current_and_past_only')
    return result,provenance


def batch(windows,device):
    output={key:torch.as_tensor(np.asarray([w[key] for w in windows]),device=device) for key in windows[0] if key!='episode_id'}
    output['episode_id']=[w['episode_id'] for w in windows];require_state_only(output);return output


class TransitionBank:
    def __init__(self,directory,layouts):
        self.directory=Path(directory);self.manifest=read(self.directory/'manifest.json')
        if not self.manifest['training_only'] or set(layouts)!=set(self.manifest['data']['train_layouts']):raise ValueError('Training-only split required')
        self.layouts=set(layouts);self.allowed=set(self.manifest['episodes']);self.episodes={};self.keys=[];self.by_key={};self.refresh()

    def refresh(self):
        for path in sorted(self.directory.glob('*.npz')):
            if path.name.endswith('.tmp.npz') or path.stem in self.episodes:continue
            match=re.search(r'P25_train_(\d+)_',path.stem)
            if path.stem not in self.allowed or not match or 'layout_'+match.group(1) not in self.layouts:raise ValueError('Held-out transition')
            with np.load(path,allow_pickle=False) as source:
                d={key:source[key].copy() for key in ['codes','successors','actions','probe_valid','task']}
            self.episodes[path.stem]=d
            for t in range(len(d['actions'])):
                if d['probe_valid'][t]:
                    key=(t//10*10,t%10);self.keys.append(key);self.by_key.setdefault(key,[]).append((path.stem,t))
        return dict(episodes=len(self.episodes),states=len(self.keys),manifest_sha256=sha(self.directory/'manifest.json'))

    def sample(self,rng,size,device):
        start,offset=rng.choice(self.keys);chosen=rng.choices(self.by_key[start,offset],k=size);rows=[]
        for name,t in chosen:
            d=self.episodes[name];begin=max(0,start-10)
            rows.append(dict(task=int(d['task']),history=d['codes'][begin:start+1],
                previous_actions=np.array([int(d['actions'][j-1]) if j else 17 for j in range(begin,start+1)]),
                prefix_codes=d['codes'][start:t+1],prefix_actions=d['actions'][start:t],successors=d['successors'][t]))
        output={key:torch.as_tensor(np.asarray([r[key] for r in rows]),device=device).long() for key in rows[0]}
        output.update(start=start,offset=offset,episode_ids=[name for name,t in chosen]);require_state_only(output);return output
