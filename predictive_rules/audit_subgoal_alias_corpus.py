"""Measure task-relevant symbol aliasing using training demonstrations only."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import torch
from torch import nn
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
saved=torch.load(root/'run_grounded_rules_v9/checkpoint_001200.pt',map_location='cpu',weights_only=False)
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();m.load_state_dict(saved['model'])
windows,provenance=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])

class UnusedTransitionOutput(nn.Module):
    # The audit never computes a plan or transition. Skip this expensive,
    # independent head; confirm below that definitions/labels remain identical.
    def forward(self,context):return context.new_zeros((len(context),1,1,1))

with torch.no_grad():
    check=batch([w for w in windows if w['start']==0][:16],'cpu')
    full=m.generate(check['task'],check['history'][:,:1],check['previous_actions'][:,:1],check['history_length'],start=0)
    full_labels=m.training_targets(full,check['targets'],check['actions'])
    m.transitions=UnusedTransitionOutput()
    reduced=m.generate(check['task'],check['history'][:,:1],check['previous_actions'][:,:1],check['history_length'],start=0)
    assert torch.equal(full['definitions'],reduced['definitions']) and torch.equal(full['subgoal_logits'],reduced['subgoal_logits'])
    assert torch.equal(full_labels,m.training_targets(reduced,check['targets'],check['actions']))
    totals=Counter();groups=defaultdict(Counter);examples=[]
    for start in sorted({w['start'] for w in windows}):
        selected=[w for w in windows if w['start']==start]
        for offset in range(0,len(selected),32):
            subset=selected[offset:offset+32];d=batch(subset,'cpu');width=min(start+1,11)
            pg=m.generate(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start)
            states=m.training_targets(pg,d['targets'],d['actions']).argmax(-1).squeeze(-1)
            goals=pg['subgoal_logits'].argmax(-1).squeeze(-1)
            for i,w in enumerate(subset):
                n=int(w['goal_index']);ids=states[i,:n+1].tolist();teacher=w['actions'][:n].tolist();initial_stop=teacher[0]==0
                terminal=teacher[-1]==0;endpoint=n-1 if terminal else n
                alias=ids[0]==ids[-1];generated_alias=int(goals[i])==ids[0]
                raw_changed=bool((w['targets'][0]!=w['targets'][n]).any())
                early=next((j for j,s in enumerate(ids[:endpoint]) if s==ids[-1]),None)
                transition_counts=defaultdict(Counter)
                for j,a in enumerate(teacher):transition_counts[ids[j],a][ids[j+1]]+=1
                c=dict(windows=1,moving_windows=int(not initial_stop),initial_STOP=int(initial_stop),
                    moving_goal_equals_initial=int(not initial_stop and alias),
                    moving_goal_alias_despite_changed_image=int(not initial_stop and alias and raw_changed),
                    moving_generated_goal_equals_initial=int(not initial_stop and generated_alias),
                    generated_goal_correct=int(int(goals[i])==ids[-1]),early_goal_reappearance=int(early is not None),
                    nonterminal_windows=int(not terminal),nonterminal_goal_alias=int(not terminal and alias),
                    conflicting_transition_rows=sum(len(v)>1 for v in transition_counts.values()),
                    distinct_states=len(set(ids)),observations=len(ids))
                totals.update(c);groups[str(start)].update(c)
                if not initial_stop and alias and len(examples)<20:
                    examples.append(dict(episode=w['episode_id'],start=start,actions=teacher,states=ids,raw_changed=raw_changed))
result=dict(training_only=True,checkpoint='grounded_rules_v9_step1200',frozen_representation_used_by_v10=True,
    skipped_head_does_not_change_symbols_verified=True,data=provenance,totals=dict(totals),
    by_revision={k:dict(v) for k,v in groups.items()},examples=examples)
(root/'diagnostics/subgoal_alias_corpus_v9.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['data','examples']}))
