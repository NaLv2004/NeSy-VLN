"""Locate state aliasing in learned features versus generated definitions."""
import json,random
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
saved=torch.load(root/'run_grounded_rules_v9/checkpoint_001200.pt',map_location='cpu',weights_only=False)
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();m.load_state_dict(saved['model'])
class UnusedTransitionOutput(nn.Module):
    def forward(self,x):return x.new_zeros((len(x),1,1,1))
m.transitions=UnusedTransitionOutput() # This independent head is never used in this diagnostic.
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=[w for w in windows if w['start']==0 and w['actions'][0]!=0 and (w['supervised']|~w['action_valid']).all()]
rows=[]
with torch.no_grad():
    for offset in range(0,len(selected),32):
        subset=selected[offset:offset+32];d=batch(subset,'cpu');b=len(subset)
        pg=m.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
        features=m.observation(m.embed(d['targets']));memory=pg['ground_memory'];previous=pg['ground_last'];old=pg['ground_codes'];reps=[]
        for t in range(11):
            memory,previous,old=m.advance(memory,previous,old,features[:,t],d['targets'][:,t])
            reps.append(F.normalize(previous+m.memory_projection(memory),dim=-1))
        reps=torch.stack(reps,1);definitions=F.normalize(pg['definitions'],dim=-1)
        scores=torch.einsum('btd,bkd->btk',reps,definitions);ids=scores.argmax(-1)
        assert torch.equal(ids,m.training_targets(pg,d['targets'],d['actions']).squeeze(2).argmax(-1))
        gram=definitions@definitions.transpose(1,2);off=~torch.eye(128,dtype=torch.bool)
        top=scores.topk(2,dim=-1).values;singular=torch.linalg.svdvals(definitions)
        for i,w in enumerate(subset):
            n=int(w['goal_index']);alias=bool(ids[i,0]==ids[i,n]);distribution=singular[i].square();distribution/=distribution.sum()
            rows.append(dict(episode=w['episode_id'],alias=alias,length=n,states=int(ids[i,:n+1].unique().numel()),
                feature_endpoint_cosine=float((reps[i,0]*reps[i,n]).sum()),
                raw_visual_endpoint_cosine=float(F.cosine_similarity(features[i,0],features[i,n],dim=0)),
                prototype_cosine_mean=float(gram[i][off].mean()),prototype_cosine_max=float(gram[i][off].max()),
                prototype_effective_rank=float((-(distribution*distribution.clamp_min(1e-12).log()).sum()).exp()),
                endpoint_top_margin=float(top[i,n,0]-top[i,n,1]),
                changed_patch_fraction=float(d['targets'][i,0].ne(d['targets'][i,n]).float().mean())))
groups={}
for name,chosen in [('all',rows),('alias',[r for r in rows if r['alias']]),('different',[r for r in rows if not r['alias']])]:
    numeric=['feature_endpoint_cosine','raw_visual_endpoint_cosine','prototype_cosine_mean','prototype_cosine_max','prototype_effective_rank','endpoint_top_margin','changed_patch_fraction','states']
    groups[name]=dict(samples=len(chosen),**{key:sum(r[key] for r in chosen)/max(len(chosen),1) for key in numeric})
result=dict(training_only=True,checkpoint='grounded_rules_v9_step1200',groups=groups,rows=rows)
(root/'diagnostics/alias_geometry_v9.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
