"""Matched recorded-state comparison: chosen action versus all moving actions.

CPU only; no simulation or gradients. Uses the pinned pre-collection model.
The unnamed action IDs remain unchanged. Action0 is the terminal action of
the environment and is excluded from the main moving-action comparison.
"""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from data import ROOT,sha

def stats(rows):
    return dict(n=len(rows),**{key:sum(r[key] for r in rows)/max(1,len(rows)) for key in
        ('correct','confidence','true_probability','nll','brier')})

@torch.inference_mode()
def main():
    torch.set_num_threads(2);began=time.time()
    data=ROOT/'onpolicy_successors_v54';manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['training_only'] and manifest['no_goal_or_imitation_supervision']
    assert sha(data/'source.pt')==manifest['source_commit']['sha256']
    saved=torch.load(data/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval().requires_grad_(False)
    model.load_state_dict(saved['model']);del saved
    records=sorted((data/'counterfactual_onpolicy').glob('*.json'))
    selected=[];others=[];all_moving=[];paired=[];by_age=defaultdict(lambda: [[],[]]);reproduced=[];p_error=[]
    c=model.cfg
    for file in records:
        record=json.loads(file.read_text());assert record['source_checkpoint_sha256']==manifest['source_commit']['sha256']
        assert record['training_only'] and not record['goal_supervision']
        assert sha(file.with_suffix('.npz'))==record['output_sha256']
        with np.load(file.with_suffix('.npz'),allow_pickle=False) as archive:
            x=torch.from_numpy(archive['features'].copy())[None]
            futures=torch.from_numpy(archive['successor_features'].copy())
            recorded=torch.from_numpy(archive['recorded_actions'].copy())
        previous=torch.cat([torch.tensor([17]),recorded])[:x.shape[1]][None]
        task=torch.tensor([['chair','table'].index(record['category'])])
        contexts=model.context_sequence(x,hard(model.observe(x)),previous,task)
        for j,program in enumerate(record['programs']):
            start=program['start'];end=record['programs'][j+1]['start'] if j+1<len(record['programs']) else len(record['steps'])
            if start>=end:continue
            context=contexts[:,start];steps=record['steps'][start:end];n=len(steps)
            m=F.one_hot(torch.tensor([[s['memory'] for s in steps]]),c.memories).float()[:,:,None].expand(-1,-1,c.actions,-1)
            o=F.one_hot(torch.tensor([[s['observed'] for s in steps]]),c.observations).float()[:,:,None].expand(-1,-1,c.actions,-1)
            a=torch.arange(c.actions)[None,None].expand(1,n,-1)
            _,logp=model.transition(context,m,o,a);p=logp.exp()[0];p=p/p.sum(-1,keepdim=True)
            target=model.observe(futures[start:end][None],context).argmax(-1)[0]
            for k,step in enumerate(steps):
                chosen=step['action'];saved_p=np.array(step['proof']['root_outcome_probabilities']);saved_p/=saved_p.sum()
                reproduced.append(int(p[k,chosen].argmax())==int(saved_p.argmax()))
                p_error.append(float(np.abs(p[k,chosen].numpy()-saved_p).mean()))
                if chosen==0:continue
                position_rows=[]
                for action in range(1,c.actions):
                    prob=p[k,action];truth=int(target[k,action]);tp=float(prob[truth])
                    row=dict(correct=int(prob.argmax())==truth,confidence=float(prob.max()),true_probability=tp,
                        nll=-math.log(max(tp,1e-12)),brier=float(prob.square().sum())-2*tp+1)
                    all_moving.append(row);position_rows.append(row)
                    if action==chosen:
                        selected.append(row);by_age[k][0].append(row);chosen_row=row
                    else:others.append(row);by_age[k][1].append(row)
                paired.append(dict(correct_delta=chosen_row['correct']-sum(r['correct'] for r in position_rows)/len(position_rows),
                    true_probability_delta=chosen_row['true_probability']-sum(r['true_probability'] for r in position_rows)/len(position_rows)))
    result=dict(time=time.time(),seconds=time.time()-began,episodes=len(records),training_only=True,
        checkpoint_sha256=manifest['source_commit']['sha256'],new_simulator_rollouts=0,gradients=0,device='CPU',
        selected=stats(selected),unchosen_moving=stats(others),all_moving=stats(all_moving),
        paired_mean={key:sum(r[key] for r in paired)/len(paired) for key in paired[0]},
        by_age={str(age):dict(selected=stats(rows[0]),unchosen_moving=stats(rows[1])) for age,rows in sorted(by_age.items())},
        saved_chosen_argmax_reproduced=sum(reproduced)/len(reproduced),saved_chosen_probability_mae=sum(p_error)/len(p_error),
        limitation='Descriptive matched-state selection comparison on training-only exploration, not navigation success. CPU numerics can differ from the original CUDA collection near code boundaries.')
    out=ROOT/'diagnostic_hybrid_replay_v54/action_selection_bias.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
