"""Training-only terminal-state intervention; not a deployable oracle policy."""
import argparse,json,random,collections
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config,export_program
from predictive_nesy.graph_subgoal_interpreter import plan
from predictive_nesy.subgoal_data import load_windows,batch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
assert saved['manifest']['architecture'] in {'anonymous_visual_memory_subgoal_v6','anonymous_rollout_proof_subgoal_v8'}
model=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
directory=Path(saved['manifest']['data']['source']);windows,_=load_windows(directory,saved['manifest']['data']['train_layouts'])
source_rows=json.loads((directory/'training_rows.json').read_text());codes=np.load(directory/'visual_tokens.npy');by_episode=collections.defaultdict(list)
for i,row in enumerate(source_rows):by_episode[row['episode_id']].append(i)
selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
rows=[]
with torch.no_grad():
    for w in selected:
        d=batch([w],'cpu');p=model.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
        observed=model.training_targets(p,d['targets'],d['actions']).argmax(-1);initial=observed[0,0].numpy()
        full=torch.from_numpy(codes[by_episode[w['episode_id']]].astype(np.int64))[None]
        # This is actual future data, used only to diagnose whether an extra
        # learned terminal goal could help. It is NEVER passed to an evaluator.
        terminal=model.encode_sequence(p,full)[0][0,-1].argmax().reshape(1).numpy()
        program=export_program(p);old=plan(program,initial);new=plan(program,initial,terminal_goal=terminal)
        n=int(w['goal_index']);teacher=[int(x) for x in w['actions'][:n]]
        rows.append(dict(episode=w['episode_id'],teacher=teacher,old_plan=old['actions'],diagnostic_plan=new['actions'],
            subgoal=program.subgoal.tolist(),actual_terminal_code=terminal.tolist(),initial=initial.tolist(),
            teacher_terminal=0 in teacher))
def stats(key):
    return dict(first_action_correct=sum(bool(r[key]) and r[key][0]==r['teacher'][0] for r in rows),
        complete_teacher_sequence_correct=sum(r[key]==r['teacher'] for r in rows),
        no_plan=sum(not r[key] for r in rows),
        nonterminal_STOP=sum(not r['teacher_terminal'] and bool(r[key]) and r[key][-1]==0 for r in rows),
        terminal_STOP=sum(r['teacher_terminal'] and bool(r[key]) and r[key][-1]==0 for r in rows))
result=dict(checkpoint=a.checkpoint,training_only=True,oracle_intervention_not_deployable=True,samples=len(rows),
    nonterminal_windows=sum(not r['teacher_terminal'] for r in rows),old=stats('old_plan'),task_goal_intervention=stats('diagnostic_plan'),rows=rows)
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
