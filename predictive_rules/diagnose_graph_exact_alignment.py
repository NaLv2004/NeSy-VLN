"""Training-only comparison of relaxed proof supervision and exact execution."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.graph_subgoal_model import GraphSubgoalModel,Config,straight_through
from predictive_nesy.graph_subgoal_training import proof_log_support,state_support
from predictive_nesy.graph_subgoal_interpreter import export_program,plan
from predictive_nesy.subgoal_data import load_windows,batch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(23);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=GraphSubgoalModel(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
candidates=[w for w in windows if w['start']==0 and w['supervised'][0]];selected=random.Random(23).sample(candidates,128)
rows=[];began=time.time()
with torch.no_grad():
    for offset in range(0,len(selected),16):
        subset=selected[offset:offset+16];d=batch(subset,'cpu');n=len(subset)
        prog=model.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
        state=straight_through(model.ground(prog,d['targets'][:,0]).squeeze(1));goal=straight_through(prog['subgoal_logits'].squeeze(1).softmax(-1))
        support=state_support(state,proof_log_support(prog,goal,10)[10]);soft=support.argmax(-1)
        for i,w in enumerate(subset):
            route=plan(export_program(prog,i),np.array([state[i].argmax().item()]));action=route['actions'][0] if route['found'] else -1
            rows.append(dict(episode=w['episode_id'],expected=int(w['actions'][0]),soft=int(soft[i]),hard=action,
                found=route['found'],plan_length=len(route['actions']),final_stop=bool(route['actions'] and route['actions'][-1]==0)))
count=len(rows);result=dict(training_only=True,checkpoint=a.checkpoint,update=saved['update'],samples=count,
    training_first_action_accuracy=sum(r['soft']==r['expected'] for r in rows)/count,
    exact_first_action_accuracy=sum(r['hard']==r['expected'] for r in rows)/count,
    training_execution_agreement=sum(r['soft']==r['hard'] for r in rows)/count,
    plan_found_rate=sum(r['found'] for r in rows)/count,
    immediate_stop=sum(r['hard']==0 for r in rows),demonstrated_immediate_stop=sum(r['expected']==0 for r in rows),
    seconds=time.time()-began,rows=rows)
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
