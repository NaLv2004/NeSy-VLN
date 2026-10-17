"""Training-only read-only attribution; no new simulator episodes or updates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import torch
from joint_model import JointObservedWorld, Config, branch_states
from train import ROOT, Windows, prepare, observed_objective
from predictive_nesy.subgoal_data import batch, TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);rng=random.Random(2916)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=JointObservedWorld(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
    corpus=Windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
    features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cpu')
    rows=[corpus.window(*k) for k in rng.choices(corpus.by_start[4],k=4)];data=batch(rows,'cpu')
    p=prepare(model,data,features,rows)
    _,_,_,parts=observed_objective(model,data,p,return_parts=True)
    watched={k:[v for n,v in model.named_parameters() if n.startswith(k)] for k in
             ['observation.','tokens.','definitions.','visual_memory.','definition_rules.','subgoal.']}
    gradient={}
    for loss_name in ['proof','dynamics','goal']:
        parameters=[v for group in watched.values() for v in group]
        gs=torch.autograd.grad(parts[loss_name],parameters,retain_graph=True,allow_unused=True)
        offset=0;result={}
        for k,group in watched.items():
            g=gs[offset:offset+len(group)];offset+=len(group)
            result[k]=sum(float(x.norm()) for x in g if x is not None)
        gradient[loss_name]=result
    del p,parts
    results=[]
    with torch.no_grad():
        for record in saved['manifest']['transition_banks']:
            bank=TransitionBank(record['path'],saved['manifest']['data']['train_layouts'])
            total=correct=unique_states=unique_images=0;state_pairs=Counter();action_counts=Counter();pred_counts=Counter()
            for _ in range(4):
                d=bank.sample(rng,4,'cpu');raw,available=features.transitions(d,bank.directory)
                lengths=torch.full((4,),d['history'].shape[1],dtype=torch.long)
                p=model.prepare(d['task'],d['history'],d['previous_actions'],lengths,start=d['start'],boundary_features=raw,boundary_available=available)
                before,after=branch_states(model,p,d);s0=before.argmax(-1);s1=after.argmax(-1)
                z0=torch.nn.functional.one_hot(s0,512).float()[:,None].expand(-1,17,-1)
                z1=torch.nn.functional.one_hot(s1,512).float();pred=model.inverse_from_discrete(z0,z1).argmax(-1)
                changed=d['successors'].ne(d['prefix_codes'][:,-1,None]).any(-1)
                for j in range(4):
                    ids=changed[j].nonzero().flatten().tolist()
                    total+=len(ids);unique_states+=len({int(s1[j,a]) for a in ids})
                    unique_images+=len({tuple(d['successors'][j,a].tolist()) for a in ids})
                    for a in ids:
                        correct+=int(pred[j,a]==a);action_counts[a]+=1;pred_counts[int(pred[j,a])]+=1
                        state_pairs[int(s0[j]),int(s1[j,a]),a]+=1
            groups={}
            for (s,t,a),n in state_pairs.items():groups.setdefault((s,t),Counter())[a]+=n
            results.append(dict(bank=bank.directory.name,probed_states=16,changed_action_examples=total,
                inverse_accuracy=correct/max(total,1),per_context_hard_ID_discrimination_upper_bound=unique_states/max(total,1),
                per_context_image_discrimination_upper_bound=unique_images/max(total,1),
                pooled_pair_discrimination_upper_bound=sum(max(c.values()) for c in groups.values())/max(total,1),
                predictions=dict(pred_counts),labels=dict(action_counts)))
    result=dict(checkpoint=args.checkpoint,update=saved['update'],training_only=True,optimizer_updates=0,
                gradient_by_loss=gradient,branches=results)
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
