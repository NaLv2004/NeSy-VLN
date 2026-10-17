"""Exact, count-weighted audit on saved training transitions only.

No optimizer, new rollout, held-out task, or policy modification. The same
categorical observation labels are evaluated under each retained program.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from planning_model import PlanningObservedWorld, Config
from joint_model import branch_states
from train import Windows, prepare, ROOT
from predictive_nesy.subgoal_data import batch, TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures
from predictive_nesy.data import sha


def ratios(counts):
    result=dict(counts)
    for name,denominator in [('transition_correct','transitions'),
                             ('generated_goal_correct','windows'),
                             ('target_endpoint_alias_before_end','nonterminal_prefix_states'),
                             ('generated_goal_alias_before_end','nonterminal_prefix_states'),
                             ('stop_edge_to_target_before_end','nonterminal_prefix_states'),
                             ('stop_edge_to_generated_goal_before_end','nonterminal_prefix_states')]:
        if counts[denominator]:result[name+'_rate']=counts[name]/counts[denominator]
    return result


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--gpu',type=int,default=6);args=ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    used=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used<256,f'GPU{args.gpu} is occupied; do not disturb other work'
    torch.set_num_threads(2);began=time.time()
    saved=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    commit=json.loads(Path(args.checkpoint).with_suffix('.COMMITTED.json').read_text())
    assert sha(args.checkpoint)==commit['sha256']
    model=PlanningObservedWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    corpus=Windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
    features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cuda')
    totals=Counter();by_age=defaultdict(Counter);by_boundary=defaultdict(Counter);goal_ids=Counter()
    for start,keys in corpus.by_start.items():
        for offset in range(0,len(keys),32):
            rows=[corpus.window(*key) for key in keys[offset:offset+32]]
            data=batch(rows,'cuda');p=prepare(model,data,features,rows)
            labels=model.training_targets(p,data['targets'],data['actions']).argmax(-1).squeeze(-1)
            index=torch.arange(len(rows),device='cuda');h=min(10,32-start)
            actual_goal=labels[index,data['goal_index']];goal=p['subgoal_logits'].argmax(-1).flatten()
            probability=model.rows(p,labels[:,:h]);table=probability.argmax(-1)
            predictions=table.gather(-1,data['actions'][:,:h,None]).squeeze(-1)
            valid=data['action_valid'][:,:h]
            moving=valid & data['actions'][:,:h].ne(0)
            correct=predictions.eq(labels[:,1:h+1])
            target_alias=labels[:,:h].eq(actual_goal[:,None]) & moving
            goal_alias=labels[:,:h].eq(goal[:,None]) & moving
            stop_target=table[:,:,0].eq(actual_goal[:,None]) & moving
            stop_generated=table[:,:,0].eq(goal[:,None]) & moving
            count=Counter(windows=len(rows),transitions=int(valid.sum()),transition_correct=int((correct&valid).sum()),
                generated_goal_correct=int(goal.eq(actual_goal).sum()),nonterminal_prefix_states=int(moving.sum()),
                target_endpoint_alias_before_end=int(target_alias.sum()),generated_goal_alias_before_end=int(goal_alias.sum()),
                stop_edge_to_target_before_end=int(stop_target.sum()),stop_edge_to_generated_goal_before_end=int(stop_generated.sum()))
            totals.update(count);by_boundary[start].update(count);goal_ids.update(goal.cpu().tolist())
            for age in range(h):
                by_age[age].update(transitions=int(valid[:,age].sum()),transition_correct=int((correct[:,age]&valid[:,age]).sum()),
                    nonterminal_prefix_states=int(moving[:,age].sum()),target_endpoint_alias_before_end=int(target_alias[:,age].sum()),
                    generated_goal_alias_before_end=int(goal_alias[:,age].sum()),stop_edge_to_target_before_end=int(stop_target[:,age].sum()),
                    stop_edge_to_generated_goal_before_end=int(stop_generated[:,age].sum()))
        print(json.dumps(dict(stage='demonstrations',completed_boundary=start,transitions=totals['transitions'])),flush=True)
    banks=[];branch_total=Counter()
    for manifest in saved['manifest']['transition_banks']:
        bank=TransitionBank(manifest['path'],saved['manifest']['data']['train_layouts']);counts=Counter()
        for (start,offset),keys in sorted(bank.by_key.items()):
            for begin_index in range(0,len(keys),32):
                selected=keys[begin_index:begin_index+32];items=[]
                for name,t in selected:
                    d=bank.episodes[name];begin=max(0,start-10)
                    items.append(dict(task=int(d['task']),history=d['codes'][begin:start+1],
                        previous_actions=np.asarray([int(d['actions'][j-1]) if j else 17 for j in range(begin,start+1)]),
                        prefix_codes=d['codes'][start:t+1],successors=d['successors'][t]))
                data={k:torch.tensor(np.asarray([r[k] for r in items]),device='cuda').long() for k in items[0]}
                data.update(start=start,episode_ids=[name for name,t in selected])
                raw,available=features.transitions(data,bank.directory)
                p=model.prepare(data['task'],data['history'],data['previous_actions'],
                    torch.full((len(items),),data['history'].shape[1],device='cuda',dtype=torch.long),
                    start=start,boundary_features=raw,boundary_available=available)
                before,after=branch_states(model,p,data);source=before.argmax(-1);target=after.argmax(-1)
                predicted=model.rows(p,source[:,None])[:,0].argmax(-1)
                counts.update(positions=len(items),transitions=target.numel(),transition_correct=int(predicted.eq(target).sum()))
                by_age['branch_'+str(offset)].update(transitions=target.numel(),transition_correct=int(predicted.eq(target).sum()))
        banks.append(dict(bank=str(bank.directory),**ratios(counts)));branch_total.update(counts)
        print(json.dumps(banks[-1]),flush=True)
    result=dict(checkpoint=args.checkpoint,checkpoint_sha256=commit['sha256'],training_only=True,optimizer_updates=0,new_rollouts=0,
        all_arbitrary_boundary_windows=True,demonstrations=ratios(totals),branches=ratios(branch_total),banks=banks,
        by_age={str(k):ratios(v) for k,v in by_age.items()},by_boundary={str(k):ratios(v) for k,v in by_boundary.items()},
        common_generated_goals_external=[(k+1,v) for k,v in goal_ids.most_common(10)],seconds=time.time()-began)
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['by_age','by_boundary','banks']}))


if __name__=='__main__':main()
