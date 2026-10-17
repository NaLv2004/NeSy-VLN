"""Training-only diagnosis of spurious short symbolic goal proofs."""
import json,time,collections,argparse
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.subgoal_model import SubgoalRuleModel,Config,straight_through
from predictive_nesy.subgoal_interpreter import export_program,plan
from predictive_nesy.subgoal_training import transition
from predictive_nesy.subgoal_data import TransitionBank
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(root/'run_subgoal_v1/checkpoint_001200.pt'));p.add_argument('--out',default=str(root/'run_subgoal_v1/training_short_goal_diagnosis.json'));args=p.parse_args()
saved=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
model=SubgoalRuleModel(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
bank=TransitionBank(root/'counterfactual_anonymous_v5',saved['manifest']['data']['train_layouts'])
keys=bank.by_key[(0,0)][:96];counts=collections.Counter();examples=[];began=time.time()
with torch.no_grad():
    for name,t in keys:
        d=bank.episodes[name];initial=torch.tensor(d['codes'][0:1],device='cuda',dtype=torch.long)
        p=model.generate(torch.tensor([int(d['task'])],device='cuda'),initial[:,None],torch.tensor([[17]],device='cuda'),torch.tensor([1],device='cuda'),start=0)
        state=model.ground(p,initial).argmax(-1)[0].cpu().numpy();route=plan(export_program(p),state)
        counts['episodes']+=1;counts['found']+=route['found']
        if route['found'] and route['actions'][-1]==0 and len(route['actions'])<=2:
            a=route['actions'];sequence=[initial[0]]
            if len(a)==2:sequence.append(torch.tensor(d['successors'][0,a[0]],device='cuda',dtype=torch.long))
            sequence.append(sequence[-1]);codes=torch.stack(sequence)[None];actions=torch.tensor([a],device='cuda')
            actual=model.training_targets(p,codes,actions)[0,-1].argmax(-1).cpu().numpy()
            match=bool(np.array_equal(actual,route['subgoal']));counts['short_stop_proof']+=1;counts['short_proof_actual_match']+=match
            examples.append(dict(episode=name,actions=a,goal=route['subgoal'],actual_post_action_state=actual.tolist(),match=match))
result=dict(training_only=True,checkpoint=args.checkpoint,checkpoint_update=saved['update'],counts=counts,seconds=time.time()-began,examples=examples)
Path(args.out).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='examples'}),flush=True)
