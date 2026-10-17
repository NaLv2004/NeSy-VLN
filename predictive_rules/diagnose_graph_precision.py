"""Compare the actual AMP training alphabet with FP32 inference on train data."""
import argparse,json,random,time
from pathlib import Path
import torch
import numpy as np
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.graph_subgoal_interpreter import export_program,plan
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3);torch.backends.cuda.matmul.allow_tf32=True
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=VisualGraphSubgoalModel(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
rows=[]
with torch.no_grad():
    for offset in range(0,128,16):
        subset=selected[offset:offset+16];d=batch(subset,'cuda');variants={}
        for name,enabled in [('fp32',False),('bf16',True)]:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
                p=model.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
                observed=model.training_targets(p,d['targets'],d['actions']).squeeze(2).argmax(-1)
            variants[name]=(p,observed)
        for i,w in enumerate(subset):
            row={'episode':w['episode_id'],'first_teacher_action':int(w['actions'][0]),'rule_argmax_agreement':float((variants['fp32'][0]['transitions'][i].argmax(-1)==variants['bf16'][0]['transitions'][i].argmax(-1)).float().mean())}
            for name,(p,observed) in variants.items():
                program=export_program(p,i);initial=int(observed[i,0]);n=int(w['goal_index']);route=plan(program,np.array([initial]))
                predicted=int(program.transitions[initial,int(w['actions'][0])].argmax())
                row[name]=dict(initial=initial,next_observed=int(observed[i,1]),predicted_next=predicted,
                    generated_goal=int(program.subgoal[0]),teacher_goal=int(observed[i,n]),first_action=route['actions'][0] if route['found'] else -1)
            rows.append(row)
result=dict(training_only=True,checkpoint=a.checkpoint,samples=len(rows),
    initial_state_agreement=sum(r['fp32']['initial']==r['bf16']['initial'] for r in rows)/128,
    next_state_label_agreement=sum(r['fp32']['next_observed']==r['bf16']['next_observed'] for r in rows)/128,
    generated_goal_agreement=sum(r['fp32']['generated_goal']==r['bf16']['generated_goal'] for r in rows)/128,
    rule_argmax_agreement=sum(r['rule_argmax_agreement'] for r in rows)/128,
    modes={name:dict(first_action_accuracy=sum(r[name]['first_action']==r['first_teacher_action'] for r in rows)/128,
        one_step_accuracy=sum(r[name]['predicted_next']==r[name]['next_observed'] for r in rows)/128,
        unique_initial_states=len({r[name]['initial'] for r in rows}),
        initial_state_equals_teacher_goal=sum(r[name]['initial']==r[name]['teacher_goal'] for r in rows)/128,
        immediate_stop_count=sum(r[name]['first_action']==0 for r in rows)) for name in ['fp32','bf16']},rows=rows)
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
