"""Training-only counterfactual audit of anonymous future-goal supervision."""
import argparse,json,random,time
from pathlib import Path
import torch
from predictive_nesy.graph_subgoal_model import GraphSubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=GraphSubgoalModel(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=random.Random(744).sample([w for w in windows if w['start']==0 and w['goal_index']>=4 and w['supervised'].all()],128)
rows=[];began=time.time()
with torch.no_grad():
    for offset in range(0,len(selected),16):
        d=batch(selected[offset:offset+16],'cpu');b=len(d['task']);indices=torch.arange(b)
        p=model.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
        original=model.training_targets(p,d['targets'],d['actions']).squeeze(2).argmax(-1)
        still=d['targets'][:,:1].expand_as(d['targets'])
        unchanged_images=model.training_targets(p,still,d['actions']).squeeze(2).argmax(-1)
        erased_actions=model.training_targets(p,d['targets'],torch.full_like(d['actions'],17)).squeeze(2).argmax(-1)
        stop_labels=model.training_targets(p,d['targets'][:,:1].expand(-1,2,-1),torch.zeros(b,1,dtype=torch.long)).squeeze(2).argmax(-1)[:,1]
        goal=original[indices,d['goal_index']];unchanged=unchanged_images[indices,d['goal_index']];erased=erased_actions[indices,d['goal_index']]
        for i,w in enumerate(selected[offset:offset+16]):
            rows.append(dict(episode=w['episode_id'],length=int(w['goal_index']),initial=int(original[i,0]),goal=int(goal[i]),
                repeated_initial_image_goal=int(unchanged[i]),erased_action_history_goal=int(erased[i]),
                immediate_stop_grounded_state=int(stop_labels[i]),generated_goal=int(p['subgoal_logits'][i].argmax())))
count=len(rows)
result=dict(training_only=True,checkpoint=a.checkpoint,update=saved['update'],samples=count,
    goal_unchanged_when_all_future_images_replaced=sum(r['goal']==r['repeated_initial_image_goal'] for r in rows)/count,
    goal_unchanged_when_future_action_tags_erased=sum(r['goal']==r['erased_action_history_goal'] for r in rows)/count,
    immediate_stop_matches_teacher_future_state=sum(r['goal']==r['immediate_stop_grounded_state'] for r in rows)/count,
    immediate_stop_matches_generated_goal=sum(r['generated_goal']==r['immediate_stop_grounded_state'] for r in rows)/count,
    unique_teacher_goal_states=len({r['goal'] for r in rows}),seconds=time.time()-began,rows=rows)
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
