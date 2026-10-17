"""Read-only training-data interventions; never an executable goal fallback."""
import argparse,json,random,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from predictive_nesy.graph_subgoal_model import GraphSubgoalModel,Config,straight_through
from predictive_nesy.graph_subgoal_interpreter import export_program,Program,plan
from predictive_nesy.subgoal_data import load_windows,batch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
if saved['manifest']['architecture']=='anonymous_visual_graph_subgoal_v5':
    from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel as GraphSubgoalModel
elif saved['manifest']['architecture'] in {'anonymous_visual_memory_subgoal_v6','anonymous_rollout_proof_subgoal_v8'}:
    from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel as GraphSubgoalModel,export_program
elif saved['manifest']['architecture']=='anonymous_definition_bound_subgoal_v7':
    from predictive_nesy.definition_bound_model import DefinitionBoundModel as GraphSubgoalModel,export_program
elif saved['manifest']['architecture']=='anonymous_definition_conditioned_world_v12':
    from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld as GraphSubgoalModel,export_program
model=GraphSubgoalModel(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
rows=[];began=time.time()
with torch.no_grad():
    for offset in range(0,128,16):
        subset=selected[offset:offset+16];d=batch(subset,'cpu')
        p=model.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
        actual=model.training_targets(p,d['targets'],d['actions']).squeeze(2).argmax(-1)
        for i,w in enumerate(subset):
            generated=export_program(p,i);initial=int(actual[i,0]);n=int(w['goal_index']);teacher_goal=int(actual[i,n])
            route=plan(generated,np.array([initial]));oracle=Program(generated.transitions,generated.definitions,np.array([teacher_goal]),0)
            alternate=plan(oracle,np.array([initial]));state=initial;matches=[]
            for t in range(n):
                state=int(generated.transitions[state,int(w['actions'][t])].argmax());matches.append(state==int(actual[i,t+1]))
            rows.append(dict(episode=w['episode_id'],length=n,first_teacher_action=int(w['actions'][0]),initial=initial,
                teacher_window_contains_STOP=bool(0 in w['actions'][:n]),
                teacher_actions=[int(x) for x in w['actions'][:n]],
                observed_state_sequence=[int(x) for x in actual[i,:n+1]],
                generated_goal=int(generated.subgoal[0]),teacher_goal=teacher_goal,
                generated_plan=route['actions'],diagnostic_teacher_goal_plan=alternate['actions'],
                predicted_teacher_endpoint=state,predicted_matches_observed=matches,
                planned_stop_changes_state=bool(route['actions'] and route['actions'][-1]==0 and route['states'][-2]!=route['states'][-1]),
                stop_rule_identity_fraction=float((generated.transitions[:,0].argmax(-1)==np.arange(model.cfg.states)).mean()),
                stop_rule_identity_on_observed=float((generated.transitions[actual[i,:n+1].numpy(),0].argmax(-1)==actual[i,:n+1].numpy()).mean())))
def first(r,key):return r[key][0] if r[key] else -1
result=dict(training_only=True,checkpoint=a.checkpoint,update=saved['update'],samples=len(rows),
    generated_first_action_accuracy=sum(first(r,'generated_plan')==r['first_teacher_action'] for r in rows)/len(rows),
    teacher_goal_first_action_accuracy=sum(first(r,'diagnostic_teacher_goal_plan')==r['first_teacher_action'] for r in rows)/len(rows),
    generated_goal_accuracy=sum(r['generated_goal']==r['teacher_goal'] for r in rows)/len(rows),
    teacher_path_endpoint_accuracy=sum(r['predicted_teacher_endpoint']==r['teacher_goal'] for r in rows)/len(rows),
    plans_ending_in_STOP=sum(bool(r['generated_plan']) and r['generated_plan'][-1]==0 for r in rows),
    planned_stop_changes_state=sum(r['planned_stop_changes_state'] for r in rows),
    stop_rule_identity_fraction=float(np.mean([r['stop_rule_identity_fraction'] for r in rows])),
    stop_rule_identity_on_observed=float(np.mean([r['stop_rule_identity_on_observed'] for r in rows])),
    generated_first_action_counts=dict(Counter(first(r,'generated_plan') for r in rows)),
    teacher_first_action_counts=dict(Counter(r['first_teacher_action'] for r in rows)),
    premature_stop_on_nonterminal_windows=sum(not r['teacher_window_contains_STOP'] and bool(r['generated_plan']) and r['generated_plan'][-1]==0 for r in rows),
    nonterminal_windows=sum(not r['teacher_window_contains_STOP'] for r in rows),
    predicted_observed_agreement_by_offset=[dict(offset=t+1,total=sum(len(r['predicted_matches_observed'])>t for r in rows),
        matched=sum(r['predicted_matches_observed'][t] for r in rows if len(r['predicted_matches_observed'])>t)) for t in range(10)],
    seconds=time.time()-began,rows=rows)
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
