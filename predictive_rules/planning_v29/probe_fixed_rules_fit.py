"""Isolated capacity diagnostic on training-only fixed categorical targets.

The resulting diagnostic model is not deployed or evaluated on development.
All source checkpoints stay immutable. No simulator or new trajectories.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import time

import torch
from planning_model import PlanningObservedWorld, Config
from joint_model import branch_states
from train import ROOT
from predictive_nesy.subgoal_data import TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);rng=random.Random(3017);torch.manual_seed(3017)
    used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used<256,f'GPU7 occupied: {used}'
    saved=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    model=PlanningObservedWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cuda')
    examples=[]
    with torch.no_grad():
        for record in saved['manifest']['transition_banks']:
            bank=TransitionBank(record['path'],saved['manifest']['data']['train_layouts'])
            eligible=[name for name,d in bank.episodes.items() if len(d['actions'])>=5]
            for name in rng.sample(eligible,min(4,len(eligible))):
                d=bank.episodes[name]
                for start in list(range(0,len(d['actions']),10))[:2]:
                    if not d['probe_valid'][start]:continue
                    begin=max(0,start-10)
                    history=torch.tensor(d['codes'][None,begin:start+1],device='cuda').long()
                    previous=torch.tensor([[int(d['actions'][j-1]) if j else 17 for j in range(begin,start+1)]],device='cuda')
                    base=dict(task=torch.tensor([int(d['task'])],device='cuda'),history=history,previous_actions=previous,start=start,episode_ids=[name])
                    raw,available=features.transitions(base,bank.directory)
                    p=model.prepare(base['task'],history,previous,torch.tensor([history.shape[1]],device='cuda'),start=start,boundary_features=raw,boundary_available=available)
                    sources=[];targets=[]
                    for t in range(start,min(start+10,len(d['actions']))):
                        if not d['probe_valid'][t]:continue
                        item=dict(base,prefix_codes=torch.tensor(d['codes'][None,start:t+1],device='cuda').long(),successors=torch.tensor(d['successors'][None,t],device='cuda').long())
                        a,b=branch_states(model,p,item);sources.append(a.argmax(-1));targets.append(b.argmax(-1))
                    if sources:
                        examples.append(dict(episode=name,bank=bank.directory.name,start=start,
                            prepared={k:p[k].detach().clone() if p[k] is not None else None for k in ['definitions','rule_context','rule_visual','rule_available']},
                            sources=torch.stack(sources,1),targets=torch.stack(targets,1)))
    del features,bank,saved
    for n,p in model.named_parameters():p.requires_grad_(n.startswith(('definition_rules.','rule_evidence.')))
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    # Split whole episodes inside the training layouts; no development data.
    names=sorted({x['episode'] for x in examples});rng.shuffle(names);held=set(names[::4])
    fit=[x for x in examples if x['episode'] not in held];check=[x for x in examples if x['episode'] in held]
    def loss_for(x):
        p=model.rows(x['prepared'],x['sources'])
        nll=-p.gather(-1,x['targets'][...,None]).squeeze(-1).clamp_min(1e-8).log()
        return nll.mean(),int((p.argmax(-1)==x['targets']).sum()),x['targets'].numel()
    @torch.no_grad()
    def measure(rows):
        losses=[];correct=total=0
        for x in rows:
            loss,n,m=loss_for(x);losses.append(float(loss)*m);correct+=n;total+=m
        return dict(correct=correct,total=total,accuracy=correct/total,nll=sum(losses)/total)
    records=[];started=time.time()
    for update in range(601):
        if update in [0,100,300,600]:
            row=dict(update=update,fit=measure(fit),held_training_episodes=measure(check),seconds=time.time()-started)
            records.append(row);print(json.dumps(row),flush=True)
        if update==600:break
        optimizer.zero_grad(set_to_none=True)
        selected=rng.choices(fit,k=4)
        loss=sum(loss_for(x)[0] for x in selected)/len(selected)
        loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step()
    result=dict(checkpoint=args.checkpoint,training_only=True,development_used=False,new_rollouts=0,
                isolated_optimizer_updates=600,lr=1e-4,representation_and_labels_frozen=True,
                runtime_policy_unchanged=True,source_checkpoint_unchanged=True,
                fit_programs=len(fit),held_training_programs=len(check),records=records)
    Path(args.out).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
