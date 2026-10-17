"""One isolated CPU-only categorical rule-learning experiment.

Encoder, definitions, grounded/predicted memories and targets are frozen.
Fitting and holdout episodes are disjoint and all belong to training layouts.
No new simulator rollouts, no validation data, and no runtime changes.
"""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SRC=ROOT/'src_temporal_symbols_v55'
sys.path.insert(0,str(SRC))
from model import FactoredWorld, Config, hard
from data import sha
sys.path.insert(0,str(Path(__file__).resolve().parent))
from definition_rule import DefinitionRule, definition_inputs


def report(logp,target,current):
    probability=logp.exp();prediction=probability.argmax(-1)
    moving=torch.arange(target.shape[1])[None].expand_as(target)>0
    changed=(target!=current)&moving
    return dict(outcomes=target.numel(),accuracy=float((prediction==target).float().mean()),
        moving_accuracy=float((prediction[moving]==target[moving]).float().mean()),
        changed_accuracy=float((prediction[changed]==target[changed]).float().mean()),
        nll=float(F.nll_loss(logp.flatten(0,1),target.flatten())),
        confidence=float(probability.max(-1).values.mean()),
        persistence_baseline=float((target==current).float().mean()))


def main():
    torch.set_num_threads(2);torch.manual_seed(561017)
    rng=random.Random(561017);start=time.time()
    out=ROOT/'diagnostic_definition_rules_v56';out.mkdir(exist_ok=True)
    source=ROOT/'diagnostic_temporal_symbols_v55/source.pt'
    commit=json.loads(source.with_suffix('.COMMITTED.json').read_text());assert sha(source)==commit['sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval().requires_grad_(False)
    model.load_v54(saved['model']);del saved
    root=ROOT/'onpolicy_successors_v54';manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['training_only'] and manifest['no_goal_or_imitation_supervision']
    paths=sorted((root/'counterfactual_onpolicy').glob('*.json'))
    buckets=defaultdict(list)
    for file in paths:
        row=json.loads(file.read_text())
        assert row['scene'] in manifest['train_layouts'] and row['training_only']
        # Keep all domains/distances/categories in the fit/holdout partition.
        key=(row.get('domain'),row.get('distance_group'),row['category'])
        buckets[key].append(file)
    holdout=set()
    for key,values in sorted(buckets.items(),key=lambda p:str(p[0])):
        rng.shuffle(values);holdout.update(values[:max(1,len(values)//5)])
    assert len(holdout)>=12
    cached={name:[] for name in ('fit','holdout')};records={name:[] for name in cached}
    codebook_changes=[];c=model.cfg
    with torch.inference_mode():
        for file in paths:
            meta=json.loads(file.read_text());assert sha(file.with_suffix('.npz'))==meta['output_sha256']
            with np.load(file.with_suffix('.npz'),allow_pickle=False) as a:
                x=torch.from_numpy(a['features'].copy())[None]
                futures=torch.from_numpy(a['successor_features'].copy())
                actions=torch.from_numpy(a['recorded_actions'].copy())
            previous=torch.cat([torch.tensor([17]),actions])[:x.shape[1]][None]
            task=torch.tensor([['chair','table'].index(meta['category'])])
            contexts=model.context_sequence(x,hard(model.observe(x)),previous,task)
            which='holdout' if file in holdout else 'fit';records[which].append(meta['episode'])
            for j,p in enumerate(meta['programs']):
                boundary=p['start'];end=meta['programs'][j+1]['start'] if j+1<len(meta['programs']) else len(meta['steps'])
                if boundary>=end:continue
                context=contexts[:,boundary]
                # Use actual deployed predicted M, not hindsight-grounded M.
                for t in range(boundary,end):
                    step=meta['steps'][t]
                    memory=F.one_hot(torch.tensor([[step['memory']]]),c.memories).float().expand(1,c.actions,-1)
                    obs=F.one_hot(torch.tensor([[step['observed']]]),c.observations).float().expand(1,c.actions,-1)
                    act=torch.arange(c.actions)[None]
                    args=definition_inputs(model,context,memory,obs,act)
                    target=model.observe(futures[t][None],context).argmax(-1)
                    _,old=model.transition(context,memory,obs,act)
                    cached[which].append(tuple(z.clone() for z in (*args,target,old)))
            if len(records[which])%20==0:print(json.dumps(dict(state='encoding',partition=which,episodes=len(records[which]),seconds=time.time()-start)),flush=True)
    tensors={key:tuple(torch.cat([r[i] for r in rows],0) for i in range(8)) for key,rows in cached.items()}
    # Clone outside inference_mode so autograd may save immutable inputs.
    tensors={key:tuple(t.clone() for t in values) for key,values in tensors.items()}
    assert not set(records['fit'])&set(records['holdout'])
    head=DefinitionRule();optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    history=[]
    def evaluate(update):
        head.eval();result={}
        with torch.no_grad():
            for key,values in tensors.items():
                predictions=[]
                for b in range(0,len(values[0]),128):predictions.append(head(*(x[b:b+128] for x in values[:6])))
                new=torch.cat(predictions)
                result[key]=report(new,values[6],values[1].argmax(-1))
                if update==0:result[key]['source']=report(values[7],values[6],values[1].argmax(-1))
        record=dict(update=update,metrics=result,seconds=time.time()-start);history.append(record)
        print(json.dumps(record),flush=True)
        (out/'progress.json').write_text(json.dumps(record,indent=2))
    evaluate(0)
    values=tensors['fit'];head.train()
    for update in range(1,801):
        chosen=torch.randint(0,len(values[0]),(128,));batch=[x[chosen] for x in values]
        optimizer.zero_grad(set_to_none=True);logp=head(*batch[:6])
        loss=F.nll_loss(logp.flatten(0,1),batch[6].flatten());loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.);assert torch.isfinite(norm)
        optimizer.step()
        if update%200==0:evaluate(update);head.train()
    with torch.no_grad():
        values=tensors['holdout'];args=[v[:16] for v in values[:6]]
        original=head(*args);permutation=torch.randperm(c.observations)
        renamed=list(args);renamed[1]=args[1][...,permutation];renamed[2]=args[2][:,permutation]
        transformed=head(*renamed)
        error=float((transformed-original[...,permutation]).abs().max())
        assert error<2e-5,error
        # Future images/labels are absent from the predictor signature.
        assert len(args)==6
    result=dict(time=time.time(),seconds=time.time()-start,source=commit,training_only=True,
        new_simulator_rollouts=0,original_dev80_used=False,finalfixed80_used=False,device='CPU',
        live_training_modified=False,fit_episodes=records['fit'],heldout_training_episodes=records['holdout'],
        positions={key:len(v[0]) for key,v in tensors.items()},history=history,
        observation_rename_equivariance_max_error=error,
        design='Shared row scorer consumes current and candidate generated definition vectors; hard symbolic premises and categorical outputs only',
        limitation='Only the new rule scorer is fitted. Codes/contexts fixed. Source checkpoint saw broader training data; this is a held-out readout probe, not independent navigation validation.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    torch.save(dict(head=head.state_dict(),source=commit,experiment=result),out/'head.pt')
    print(json.dumps({k:v for k,v in result.items() if k not in ('fit_episodes','heldout_training_episodes','history')}),flush=True)

if __name__=='__main__':main()
