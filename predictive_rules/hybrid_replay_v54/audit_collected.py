"""Offline baseline on committed current-policy training transitions only."""
import json,os,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from data import ROOT,sha
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
data=ROOT/'onpolicy_successors_v54';saved=torch.load(data/'source.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False)
model.load_state_dict(saved['model']);del saved
files=sorted((data/'counterfactual_onpolicy').glob('*.json'))
rows=[]
with torch.inference_mode():
    for file in files:
        record=json.loads(file.read_text());assert record['training_only'] and not record['goal_supervision']
        path=file.with_suffix('.npz');assert sha(path)==record['output_sha256']
        with np.load(path,allow_pickle=False) as cache:
            features=torch.tensor(cache['features'].copy(),device='cuda')[None]
            successor=torch.tensor(cache['successor_features'].copy(),device='cuda')
            actions=torch.tensor(cache['recorded_actions'].copy(),device='cuda')
        task=torch.tensor([['chair','table'].index(record['category'])],device='cuda')
        previous=torch.cat([torch.tensor([17],device='cuda'),actions])[:features.shape[1]][None]
        contexts=model.context_sequence(features,hard(model.observe(features)),previous,task)
        for program in record['programs']:
            bs=program['start'];following=[p['start'] for p in record['programs'] if p['start']>bs]
            end=min(following) if following else len(record['steps'])
            ctx=contexts[:,bs];lo,z,_=model.observation_encoding(features,ctx)
            obs=hard(lo);lm,_,_=model.memory_sequence(obs,previous,z);grounded=hard(lm)
            for t in range(bs,end):
                step=record['steps'][t];action=step['action']
                actual=model.observe(successor[t,action][None],ctx).argmax(-1)
                available_o=F.one_hot(torch.tensor([step['observed']],device='cuda'),64).float()
                grounded_m=grounded[:,t]
                _,forced=model.transition(ctx,grounded_m,available_o,torch.tensor([action],device='cuda'))
                _,on=model.transition(ctx,F.one_hot(torch.tensor([step['memory']],device='cuda'),32).float(),
                    available_o,torch.tensor([action],device='cuda'))
                # The model used to score saved evidence is exactly the pinned
                # collection model. This has no influence on chosen actions.
                p=step['proof']['root_outcome_probabilities']
                rows.append(dict(episode=record['episode'],step=t,age=t-bs,
                    recorded_observation_reproduced=int(lo[0,t].argmax())==step['observed'],
                    memory_disagreement=int(grounded_m.argmax())!=step['memory'],
                    actual_observation=int(actual),runtime_correct=int(np.argmax(p))==int(actual),
                    teacher_memory_correct=int(forced.argmax())==int(actual),
                    recorded_rule_argmax_reproduced=int(on.argmax())==int(np.argmax(p)),
                    true_probability=float(p[int(actual)]/sum(p))))
def summarize(values):
    if not values:return {'n':0}
    return dict(n=len(values),**{k:sum(float(r[k]) for r in values)/len(values) for k in
        ('recorded_observation_reproduced','memory_disagreement','runtime_correct','teacher_memory_correct','recorded_rule_argmax_reproduced','true_probability')})
result=dict(time=time.time(),episodes=len(files),training_only=True,new_simulator_rollouts=0,
    all=summarize(rows),initial=summarize([r for r in rows if r['step']==0]),
    by_age={str(a):summarize([r for r in rows if r['age']==a]) for a in range(6)},
    episode_ids=[json.loads(p.read_text())['episode'] for p in files],
    limitation='Descriptive current-policy states, not validation. Regrounded-memory counterfactual is an offline diagnostic only.')
out=ROOT/'diagnostic_hybrid_replay_v54/collected_baseline.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='episode_ids'}))
