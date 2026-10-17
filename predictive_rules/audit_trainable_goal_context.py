"""Bounded CPU probe: can learning boundary features help anonymous goals?"""
import copy,json,random,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
torch.set_num_threads(2);torch.manual_seed(4427)
saved=torch.load(root/'initialization_vocabulary_v11r1/initial.pt',map_location='cpu',weights_only=False)
model=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
class NoRules(nn.Module):
    def forward(self,x):return x.new_zeros((len(x),1,1,1))
model.transitions=NoRules();model.requires_grad_(False)
fitting=set(torch.load(root/'diagnostics/unsupervised_vocabulary_v9/centers.pt',map_location='cpu',weights_only=False)['fit_episode_ids'])
ws,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
ws=[w for w in ws if (w['supervised']|~w['action_valid']).all()]
groups={};sizes=[]
with torch.no_grad():
    for start in sorted({w['start'] for w in ws}):
        selected=[w for w in ws if w['start']==start];labels=[]
        for offset in range(0,len(selected),64):
            part=selected[offset:offset+64];d=batch(part,'cpu');width=min(start+1,11)
            p=model.generate(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start)
            target=model.training_targets(p,d['targets'],d['actions']).squeeze(2).argmax(-1)
            labels.append(target[torch.arange(len(part)),d['goal_index']])
        d=batch(selected,'cpu');d['labels']=torch.cat(labels);d['fit']=torch.tensor([w['episode_id'] in fitting for w in selected]);d['fit_ids']=d['fit'].nonzero().flatten()
        groups[start]=d;sizes.append(len(d['fit_ids']))

class GoalContext(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ['tokens','context_read','action_embed','task','history','context_norm']:
            module=copy.deepcopy(getattr(model,name));module.requires_grad_(True);setattr(self,name,module)
        self.subgoal=nn.Linear(128,128)
        self.register_buffer('offsets',model.offsets.clone())
    def forward(self,d,ids,start):
        width=min(start+1,11);x=self.tokens(d['history'][ids,:width]+self.offsets).flatten(-2)
        x=self.context_read(x)+self.action_embed(d['previous_actions'][ids,:width])
        history,_=self.history(x,self.task(d['task'][ids])[None])
        context=self.context_norm(history[torch.arange(len(ids)),d['history_length'][ids]-1])
        return self.subgoal(context)

head=GoalContext();optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-3);rng=random.Random(4427)
began=time.time();history=[]
for step in range(1,1601):
    start=rng.choices(list(groups),weights=sizes,k=1)[0];d=groups[start]
    ids=d['fit_ids'][torch.randint(len(d['fit_ids']),(128,))]
    loss=F.cross_entropy(head(d,ids,start),d['labels'][ids]);optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
    if step%200==0:
        totals={k:[0,0] for k in ['fit','heldout_training','fit_start0','heldout_training_start0']}
        with torch.no_grad():
            for start,d in groups.items():
                for offset in range(0,len(d['labels']),128):
                    ids=torch.arange(offset,min(offset+128,len(d['labels'])));correct=head(d,ids,start).argmax(-1)==d['labels'][ids]
                    for key,mask in [('fit',d['fit'][ids]),('heldout_training',~d['fit'][ids])]:
                        totals[key][0]+=int(correct[mask].sum());totals[key][1]+=int(mask.sum())
                        if start==0:totals[key+'_start0'][0]+=int(correct[mask].sum());totals[key+'_start0'][1]+=int(mask.sum())
        row=dict(update=step,seconds=time.time()-began,metrics={k:dict(correct=v[0],count=v[1],accuracy=v[0]/max(v[1],1)) for k,v in totals.items()})
        history.append(row);print(json.dumps(row),flush=True)
result=dict(training_only=True,no_development_or_final_validation=True,probe_not_deployed=True,
    altered_modules='a separate copy of boundary encoder and goal head; original grounding and vocabulary unchanged',
    label='fixed anonymous demonstrated endpoint',steps=1600,history=history,
    best_internal_holdout=max(history,key=lambda r:r['metrics']['heldout_training']['accuracy']),final=history[-1])
(root/'diagnostics/trainable_goal_context_v11.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
