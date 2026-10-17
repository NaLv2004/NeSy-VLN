"""Bounded CPU-only fit of the rule decoder on immutable train-only effects.

Episode-disjoint holdout is carved solely from the TRAINING cache. This is not
navigation evaluation, and its candidate parameters are never installed live.
"""
from collections import defaultdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from data import ROOT, sha
from model import FactoredWorld, Config


def main():
    torch.set_num_threads(3); torch.manual_seed(421017)
    began=time.time(); cache=ROOT/'onpolicy_successors_v40'
    commit=json.loads((cache/'policy.COMMITTED.json').read_text())
    assert sha(cache/'policy.pt')==commit['sha256']
    saved=torch.load(cache/'policy.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval()
    model.load_state_dict(saved['model'],strict=True); c=model.cfg
    paths=sorted((cache/'counterfactual_onpolicy').glob('*.json'))
    groups=defaultdict(list); records={}
    for path in paths:
        r=json.loads(path.read_text()); assert r['training_only'] and not r['goal_supervision']
        assert r['source_checkpoint_sha256']==commit['sha256']
        if not r['steps']:continue
        records[path]=r; groups[(r['domain'],r['category'],r['distance_group'])].append(path)
    rng=random.Random(421017); holdout=set()
    for group in groups.values():
        holdout.update(rng.sample(group,max(1,round(len(group)*.2))))
    arrays=defaultdict(list); episode_names=[]
    with torch.inference_mode():
        for path,r in records.items():
            with np.load(path.with_suffix('.npz')) as z:
                raw=z['features']; future=z['successor_features']; actions=z['recorded_actions']
            x=torch.from_numpy(raw)[None]
            o=model.observe(x).argmax(-1)
            previous=torch.tensor([[17]+actions[:len(raw)-1].tolist()])
            contexts=model.context_sequence(x,F.one_hot(o,c.observations).float(),previous,
                torch.tensor([0 if r['category']=='chair' else 1]))[0]
            flat=future.reshape(-1,49,128)
            target=torch.cat([model.observe(torch.from_numpy(flat[i:i+128])).argmax(-1) for i in range(0,len(flat),128)]).reshape(len(raw),c.actions)
            starts=[p['start'] for p in r['programs']]
            for s in r['steps']:
                t=s['step']; assert int(o[0,t])==s['observed']
                start=max(k for k in starts if k<=t)
                arrays['context'].append(contexts[start].clone())
                arrays['memory'].append(s['memory']);arrays['observation'].append(s['observed'])
                arrays['target'].append(target[t].clone());arrays['holdout'].append(path in holdout)
                episode_names.append(r['episode'])
    # Cloning outside inference mode makes ordinary tensors usable by autograd.
    context=torch.stack(arrays['context']).clone()
    memory=F.one_hot(torch.tensor(arrays['memory']),c.memories).float()
    observation=F.one_hot(torch.tensor(arrays['observation']),c.observations).float()
    target=torch.stack(arrays['target']).clone()
    test=torch.tensor(arrays['holdout']); train=torch.where(~test)[0]; valid=torch.where(test)[0]
    action=torch.arange(c.actions)[None]
    names=('definition_generator.','observation_embedding.','action_embedding.','rule.','persistence.','action_persistence.')
    model.requires_grad_(False)
    params=[p for n,p in model.named_parameters() if n.startswith(names)]
    for p in params:p.requires_grad_(True)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    def predict(indices):
        n=len(indices)
        return model.transition(context[indices],memory[indices,None].expand(-1,c.actions,-1),
            observation[indices,None].expand(-1,c.actions,-1),action.expand(n,-1))[1]
    @torch.no_grad()
    def measure(indices):
        total=correct=changed_correct=changed_count=0; loss=0.
        for batch in indices.split(64):
            pred=predict(batch); truth=target[batch]; hits=pred.argmax(-1)==truth
            changed=truth!=observation[batch].argmax(-1)[:,None]
            total+=truth.numel(); correct+=int(hits.sum()); changed_correct+=int((hits&changed).sum());changed_count+=int(changed.sum())
            loss+=float(F.nll_loss(pred.flatten(0,1),truth.flatten(),reduction='sum'))
        return dict(outcomes=total,accuracy=correct/total,nll=loss/total,
            changed_accuracy=changed_correct/max(1,changed_count),persistence_baseline=1-changed_count/total)
    history=[]
    for update in range(501):
        if update%100==0:
            row=dict(update=update,train=measure(train),holdout=measure(valid),seconds=time.time()-began)
            history.append(row);print(json.dumps(row),flush=True)
        if update==500:break
        indices=train[torch.randint(len(train),(64,))]
        optimizer.zero_grad(set_to_none=True)
        pred=predict(indices);loss=F.nll_loss(pred.flatten(0,1),target[indices].flatten())
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,5.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
    result=dict(training_only=True,cpu_only=True,source_checkpoint=commit,
        fit_episodes=sorted({episode_names[i] for i in train.tolist()}),
        heldout_training_episodes=sorted({episode_names[i] for i in valid.tolist()}),
        fit_positions=len(train),heldout_positions=len(valid),updates=500,learning_rate=1e-4,
        frozen='Visual code encoder, memory encoder/predictor, context encoder, goal generator and learned preserving-operator classes',
        learned='Rule decoder and its definition/embedding/persistence parameters only',
        history=history,seconds=time.time()-began,
        limitation='This measures next-code fitting/generalization on a frozen training corpus, not closed-loop navigation success. No candidate weights have been installed or saved as a trained policy.')
    assert not set(result['fit_episodes'])&set(result['heldout_training_episodes'])
    Path(__file__).with_name('frozen_world_fit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
