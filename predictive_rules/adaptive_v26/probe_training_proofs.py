"""CPU-only diagnostic on training demonstrations; no rollout or updates.

Compare whether generated/recorded endpoint IDs are reachable under the actual
hard table. Replacing a goal is diagnostic only, never an executed fallback.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from adaptive_model import AdaptiveObservedWorld,Config
from hard_interpreter import Program,plan
from train import ROOT,Windows,prepare
from predictive_nesy.data import read,sha
from predictive_nesy.subgoal_data import batch
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--samples',type=int,default=128);args=parser.parse_args();torch.set_num_threads(2)
    checkpoint=Path(args.checkpoint)
    assert sha(checkpoint)==read(checkpoint.with_suffix('.COMMITTED.json'))['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=AdaptiveObservedWorld(Config(**saved['model_config'])).eval()
    missing,unexpected=model.load_state_dict(saved['model'],strict=False)
    assert not unexpected and all(n.startswith(('correction.','revision.')) for n in missing)
    corpus=Windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
    features=RawBoundaryFeatures(ROOT/'prequant_visual_features_v18','cpu');rng=random.Random(6081)
    records=[];began=time.time()
    for offset in range(0,args.samples,4):
        start=0 if offset%8==0 else rng.choice(corpus.keys)[1]
        keys=rng.choices(corpus.by_start[start],k=min(4,args.samples-offset));rows=[corpus.window(*k) for k in keys]
        data=batch(rows,'cpu');p=prepare(model,data,features,rows)
        labels=model.training_targets(p,data['targets'],data['actions']).argmax(-1).squeeze(-1)
        probabilities=model.rows(p);table=probabilities.argmax(-1).numpy().astype(np.int16);confidence=probabilities.max(-1).values.numpy()
        observed=labels[:,0];corrected=model.correction.selected(p['definitions'],p['context_raw'],observed,observed).argmax(-1)
        for i,(name,t) in enumerate(keys):
            goal=int(p['subgoal_logits'][i].argmax(-1));target=int(labels[i,data['goal_index'][i]])
            observation=int(observed[i]);source=int(corrected[i]);h=min(10,32-t)
            correction=np.tile(np.arange(512,dtype=np.int16)[None],(512,1));correction[observation,observation]=source
            program=Program(table[i],confidence[i],correction,goal,t).validate()
            proof=plan(program,observation,observation,h)
            target_program=Program(table[i],confidence[i],correction,target,t).validate()
            target_proof=plan(target_program,observation,observation,h)
            identity=np.tile(np.arange(512,dtype=np.int16)[None],(512,1))
            identity_program=Program(table[i],confidence[i],identity,goal,t).validate()
            identity_proof=plan(identity_program,observation,observation,h)
            state=observation;correct=0;steps=int(data['goal_index'][i])
            for j in range(steps):
                obs=int(labels[i,j]);pred=torch.tensor([state]);obsid=torch.tensor([obs])
                context={k:v[i:i+1] for k,v in p.items() if isinstance(v,torch.Tensor)}
                premise=int(model.correction.selected(context['definitions'],context['context_raw'],pred,obsid).argmax(-1))
                state=int(table[i,premise,int(data['actions'][i,j])]);correct+=state==int(labels[i,j+1])
            records.append(dict(episode=name,start=t,goal=goal,recorded_endpoint=target,goal_correct=goal==target,
                generated_goal_reachable=proof['found'],recorded_endpoint_reachable=target_proof['found'],
                identity_goal_reachable=identity_proof['found'],
                identity_first_action_matches=identity_proof['found'] and identity_proof['actions'][0]==int(data['actions'][i,0]),
                first_action_matches=proof['found'] and proof['actions'][0]==int(data['actions'][i,0]),
                first_action_matches_with_recorded_endpoint=target_proof['found'] and target_proof['actions'][0]==int(data['actions'][i,0]),
                transition_correct=correct,transition_total=steps,corrector_preserves_observation=source==observation))
        print(json.dumps(dict(completed=len(records))),flush=True)
    flags=['goal_correct','generated_goal_reachable','recorded_endpoint_reachable','identity_goal_reachable','identity_first_action_matches','first_action_matches',
           'first_action_matches_with_recorded_endpoint','corrector_preserves_observation']
    result=dict(samples=len(records),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        training_only=True,no_rollouts=True,no_parameter_updates=True,
        rates={key:sum(r[key] for r in records)/len(records) for key in flags},
        transition_accuracy=sum(r['transition_correct'] for r in records)/sum(r['transition_total'] for r in records),
        by_boundary={key:{field:sum(r[field] for r in records if (r['start']==0)==value)/sum((r['start']==0)==value for r in records)
                          for field in flags} for key,value in [('initial',True),('later',False)]},
        seconds=time.time()-began,records=records)
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
