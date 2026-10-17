"""Small training-only probes of goal information in frozen boundary context.

No probe is a deployed policy. Future observations supply labels only. Neither
development nor final validation data is read, and no navigation is rerun.
"""
import json,time,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
torch.set_num_threads(2);torch.manual_seed(4427)
saved=torch.load(root/'initialization_vocabulary_v11r1/initial.pt',map_location='cpu',weights_only=False)
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();m.load_state_dict(saved['model'])
class NoRules(nn.Module):
    def forward(self,x):return x.new_zeros((len(x),1,1,1))
m.transitions=NoRules()
cluster=torch.load(root/'diagnostics/unsupervised_vocabulary_v9/centers.pt',map_location='cpu',weights_only=False)
fitting=set(cluster['fit_episode_ids']);windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
windows=[w for w in windows if (w['supervised']|~w['action_valid']).all()]
xs=[];ys=[];fit=[];starts=[]
with torch.no_grad():
    for start in sorted({w['start'] for w in windows}):
        selected=[w for w in windows if w['start']==start]
        for offset in range(0,len(selected),64):
            ws=selected[offset:offset+64];d=batch(ws,'cpu');width=min(11,start+1)
            p=m.generate(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start)
            labels=m.training_targets(p,d['targets'],d['actions']).squeeze(2).argmax(-1)
            xs.append(m.context_norm(p['context_raw']));ys.append(labels[torch.arange(len(ws)),d['goal_index']])
            fit.extend(w['episode_id'] in fitting for w in ws);starts.extend(start for w in ws)
x=torch.cat(xs);y=torch.cat(ys);mask=torch.tensor(fit);start0=torch.tensor(starts)==0
train_ids=mask.nonzero().flatten();began=time.time();results={}
for name,head in [('linear',nn.Linear(128,128)),('nonlinear',nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,128)))]:
    torch.manual_seed(4427);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    for step in range(1600):
        ids=train_ids[torch.randint(len(train_ids),(256,))];loss=F.cross_entropy(head(x[ids]),y[ids]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        predicted=head(x).argmax(-1);parts={}
        for label,selection in [('fit',mask),('heldout_training',~mask),('fit_start0',mask&start0),('heldout_training_start0',~mask&start0)]:
            parts[label]=dict(count=int(selection.sum()),correct=int(((predicted==y)&selection).sum()),accuracy=float((predicted[selection]==y[selection]).float().mean()))
    results[name]=parts
    print(json.dumps(dict(probe=name,results=parts,seconds=time.time()-began)),flush=True)
result=dict(training_only=True,no_development_or_final_validation=True,probe_not_deployed=True,
    input='frozen context available only at a rule-generation boundary',target='learned discrete demonstrated endpoint',
    fitting_episode_count=len(fitting),training_steps_per_probe=1600,results=results,seconds=time.time()-began)
(root/'diagnostics/goal_context_capacity_v11.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
