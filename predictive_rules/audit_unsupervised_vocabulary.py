"""Offline feasibility audit of a learned anonymous visual vocabulary.

This creates no executable controller. Clustering never uses goals, rewards,
actions, geometry, maps, or validation images. Actions only describe diagnostic
alias statistics after clustering has finished.
"""
import json,random
from collections import Counter,defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2);torch.manual_seed(1842)
saved=torch.load(root/'run_grounded_rules_v9/checkpoint_001200.pt',map_location='cpu',weights_only=False)
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();m.load_state_dict(saved['model'])
class UnusedTransitions(nn.Module):
    def forward(self,x):return x.new_zeros((len(x),1,1,1))
m.transitions=UnusedTransitions()
windows,provenance=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
episode_ids=sorted({w['episode_id'] for w in windows});random.Random(1842).shuffle(episode_ids)
fit_episodes=set(episode_ids[:int(len(episode_ids)*.8)]);records=[]
with torch.no_grad():
    for start in sorted({w['start'] for w in windows}):
        selected=[w for w in windows if w['start']==start]
        for offset in range(0,len(selected),32):
            subset=selected[offset:offset+32];d=batch(subset,'cpu');width=min(start+1,11)
            pg=m.generate(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start)
            features=m.observation(m.embed(d['targets']));memory=pg['ground_memory'];previous=pg['ground_last'];old=pg['ground_codes'];reps=[]
            for t in range(11):
                memory,previous,old=m.advance(memory,previous,old,features[:,t],d['targets'][:,t])
                reps.append(F.normalize(previous+m.memory_projection(memory),dim=-1))
            reps=torch.stack(reps,1)
            labels=torch.einsum('btd,bkd->btk',reps,F.normalize(pg['definitions'],dim=-1)).argmax(-1)
            assert torch.equal(labels,m.training_targets(pg,d['targets'],d['actions']).squeeze(2).argmax(-1))
            for i,w in enumerate(subset):
                n=int(w['goal_index']);records.append(dict(episode=w['episode_id'],fit=w['episode_id'] in fit_episodes,
                    features=reps[i,:n+1].clone(),old_labels=labels[i,:n+1].tolist(),actions=w['actions'][:n].tolist()))
    x=torch.cat([r['features'] for r in records if r['fit']]);k=128
    centers=[x[torch.randint(len(x),(1,)).item()].clone()];closest=torch.ones(len(x))*2
    for _ in range(1,k):
        closest=torch.minimum(closest,(1-x@centers[-1]).clamp_min(0))
        index=torch.multinomial(closest+1e-8,1).item();centers.append(x[index].clone())
    centers=torch.stack(centers);history=[]
    for iteration in range(40):
        scores=x@centers.T;values,assignments=scores.max(-1)
        sums=torch.zeros_like(centers);sums.index_add_(0,assignments,x);counts=torch.bincount(assignments,minlength=k)
        empty=(counts==0).nonzero().flatten()
        if len(empty):sums[empty]=x[values.argsort()[:len(empty)]]
        centers=F.normalize(sums,dim=-1);history.append(float(values.mean()))
    def summarize(part,kind):
        rows=[r for r in records if part=='all' or r['fit']==(part=='fit')];totals=Counter()
        for r in rows:
            ids=r['old_labels'] if kind=='original' else (r['features']@centers.T).argmax(-1).tolist()
            actions=r['actions'];endpoint=len(actions)-(actions[-1]==0)
            counts=defaultdict(Counter)
            for t,a in enumerate(actions):counts[ids[t],a][ids[t+1]]+=1
            totals.update(windows=1,moving_windows=int(actions[0]!=0),
                moving_endpoint_alias=int(actions[0]!=0 and ids[0]==ids[-1]),
                early_goal_alias=int(any(s==ids[-1] for s in ids[:endpoint])),
                conflicting_transition_rows=sum(len(c)>1 for c in counts.values()),
                distinct_states=len(set(ids)),observations=len(ids))
        return dict(totals)
    result=dict(training_only=True,no_validation_images=True,diagnostic_not_deployed=True,
        fitting_episodes=len(fit_episodes),held_out_training_episodes=len(episode_ids)-len(fit_episodes),
        labels_ignored_during_clustering=['actions','task','goal','reward','geometry','map'],
        learned_centers=128,feature_dimensions=256,fitting_frames=len(x),mean_cosine_by_iteration=history,
        comparison={part:{kind:summarize(part,kind) for kind in ['original','learned_vocabulary']} for part in ['fit','heldout_training','all']})
    folder=root/'diagnostics/unsupervised_vocabulary_v9';folder.mkdir(exist_ok=True)
    torch.save(dict(centers=centers,source_checkpoint=str(root/'run_grounded_rules_v9/checkpoint_001200.pt'),
        fit_episode_ids=sorted(fit_episodes),training_data=provenance,diagnostic_not_deployed=True),folder/'centers.pt')
    (folder/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='mean_cosine_by_iteration'}))
