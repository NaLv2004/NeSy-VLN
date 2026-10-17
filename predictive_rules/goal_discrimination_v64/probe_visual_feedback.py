"""Training-data-only visual diagnostic. Never loaded by a navigation policy."""
import json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
DATA=ROOT/'goal_discrimination_v64'

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);torch.manual_seed(651017);os.nice(10)
    output=DATA/'visual_feedback_probe.json'
    assert not output.exists(),'Inspect existing diagnostic instead of repeating it'
    source=torch.load(DATA/'source.pt',map_location='cpu',weights_only=False)
    held=set(source['manifest']['transition_holdout_episodes']);del source
    parts={k:dict(x=[],y=[]) for k in ('fit','holdout')}
    for path in sorted((DATA/'episodes').glob('*.json')):
        meta=json.loads(path.read_text());assert meta['training_only']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as d:
            actions=d['actions'];n=len(actions)+(not len(actions) or actions[-1]!=0)
            x=d['features'][:n].astype(np.float32).reshape(n,-1)
            task=np.full((n,1),['chair','table'].index(meta['category']),np.float32)
            # Category-specific linear readouts are a diagnostic of available
            # visual information, not symbolic states or an executable policy.
            x=np.concatenate([x*(1-task),x*task,task],axis=-1)
            p=parts['holdout' if meta['episode'] in held else 'fit']
            p['x'].append(x);p['y'].append(d['stop_success'][:n].copy())
    for d in parts.values():
        d['x']=torch.from_numpy(np.concatenate(d['x']));d['y']=torch.from_numpy(np.concatenate(d['y'])).float()
    mean=parts['fit']['x'].mean(0);scale=parts['fit']['x'].std(0).clamp_min(.02)
    for d in parts.values():d['x']=(d['x']-mean)/scale
    model=nn.Linear(len(mean),1);nn.init.zeros_(model.weight);nn.init.zeros_(model.bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.1)
    d=parts['fit'];positive=d['y'].bool();w=torch.where(positive,.5/positive.sum(),.5/(~positive).sum())
    began=time.time();rows=[]
    def score(d):
        with torch.no_grad():
            p=model(d['x']).squeeze(-1).sigmoid();y=d['y'].bool();pred=p>=.5
            negative=p[~y];cut=torch.quantile(negative,.98)
            return dict(positive=int(y.sum()),negative=int((~y).sum()),
                tpr=float(pred[y].float().mean()),fpr=float(pred[~y].float().mean()),
                tpr_at_2percent_fpr=float((p[y]>=cut).float().mean()),
                threshold_at_2percent_fpr=float(cut),
                balanced_nll=float(-.5*(p[y].clamp_min(1e-8).log().mean()+(1-p[~y]).clamp_min(1e-8).log().mean())))
    for update in range(1,151):
        optimizer.zero_grad(set_to_none=True)
        logits=model(d['x']).squeeze(-1)
        loss=(F.binary_cross_entropy_with_logits(logits,d['y'],reduction='none')*w).sum()
        loss.backward();optimizer.step()
        if update in (25,50,100,150):rows.append(dict(update=update,fit=score(parts['fit']),holdout=score(parts['holdout'])))
    result=dict(time=time.time(),seconds=time.time()-began,rows=rows,source='Frozen visual patches only plus task category',
        meaning='Diagnostic information separability, not navigation success; holdout threshold only describes the ROC curve',
        no_model_saved=True,no_new_runtime_head=True,no_dev80=True,no_finalfixed80=True,new_rollouts=0)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
