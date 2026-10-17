"""Training-only state intervention on the auxiliary visual reconstruction."""
import json,random
from pathlib import Path
import torch
import torch.nn.functional as F
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
saved=torch.load(root/'run_grounded_rules_v9/checkpoint_001200.pt',map_location='cpu',weights_only=False)
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();m.load_state_dict(saved['model'])
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
totals={name:dict(loss=0.,correct=0.,tokens=0.,changed_loss=0.,changed_correct=0.,changed_tokens=0.) for name in ['normal','constant_initial_state','reverse_states','zero_context']}
with torch.no_grad():
    for offset in range(0,128,16):
        d=batch(selected[offset:offset+16],'cpu');pg=m.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
        ids=m.training_targets(pg,d['targets'],d['actions']).squeeze(2).argmax(-1)
        hard=F.one_hot(ids,m.cfg.states).float();reverse=hard.clone()
        for i,n in enumerate(d['goal_index'].tolist()):reverse[i,:n+1]=hard[i,:n+1].flip(0)
        variants=dict(normal=(pg,hard),constant_initial_state=(pg,hard[:,:1].expand_as(hard)),reverse_states=(pg,reverse),zero_context=({**pg,'context_raw':torch.zeros_like(pg['context_raw'])},hard))
        mask=d['state_valid'][:,:,None].expand_as(d['targets']);changed=mask & d['targets'].ne(d['targets'][:,:1])
        for name,(program,state) in variants.items():
            logits=m.reconstruct_codes(program,state);ce=F.cross_entropy(logits.flatten(0,2),d['targets'].flatten(),reduction='none').reshape_as(d['targets'])
            correct=logits.argmax(-1).eq(d['targets']);r=totals[name]
            for key,v in [('loss',ce[mask].sum()),('correct',correct[mask].sum()),('tokens',mask.sum()),('changed_loss',ce[changed].sum()),('changed_correct',correct[changed].sum()),('changed_tokens',changed.sum())]:r[key]+=float(v)
for row in totals.values():
    row.update(ce=row['loss']/row['tokens'],accuracy=row['correct']/row['tokens'],
        changed_ce=row['changed_loss']/row['changed_tokens'],changed_accuracy=row['changed_correct']/row['changed_tokens'])
result=dict(training_only=True,samples=128,checkpoint='grounded_rules_v9_step1200',ablation_results=totals,
    scope='Auxiliary decoder only; no claim about navigation and no inference observations changed')
(root/'diagnostics/reconstruction_state_use_v9.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
