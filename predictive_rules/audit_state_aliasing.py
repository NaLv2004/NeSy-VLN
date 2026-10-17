"""Measure state aliasing on demonstrations without any semantic state labels."""
import argparse, json, random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.visual_graph_subgoal_model import VisualGraphSubgoalModel, Config
from predictive_nesy.subgoal_data import load_windows, batch
from predictive_nesy.graph_subgoal_interpreter import export_program, plan

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
if saved['manifest']['architecture'] in {'anonymous_visual_memory_subgoal_v6','anonymous_rollout_proof_subgoal_v8'}:
    from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel as VisualGraphSubgoalModel,export_program
elif saved['manifest']['architecture']=='anonymous_definition_bound_subgoal_v7':
    from predictive_nesy.definition_bound_model import DefinitionBoundModel as VisualGraphSubgoalModel,export_program
model=VisualGraphSubgoalModel(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
counts=Counter();records=[]
with torch.no_grad():
    for offset in range(0,len(selected),16):
        subset=selected[offset:offset+16];d=batch(subset,'cpu')
        generated=model.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0)
        ids=model.training_targets(generated,d['targets'],d['actions']).squeeze(2).argmax(-1)
        for i,w in enumerate(subset):
            n=int(w['goal_index']);actual=ids[i,:n+1].tolist();actions=w['actions'][:n].tolist()
            # Alias means one learned state and action has two different next
            # learned states in the same demonstrated segment/program.
            transitions={};conflicts=0
            for position,(s,act,t) in enumerate(zip(actual,actions,actual[1:])):
                key=(s,int(act))
                if key in transitions and transitions[key][0]!=t:
                    conflicts+=1
                    previous=transitions[key][1]
                    same_codes=np.array_equal(w['targets'][previous],w['targets'][position])
                    counts['conflicts_with_identical_current_visual_codes']+=int(same_codes)
                    counts['conflicts_with_different_current_visual_codes']+=int(not same_codes)
                transitions[key]=(t,position)
            pg=export_program(generated,i);route=plan(pg,np.array([actual[0]]))
            stops=[j for j,act in enumerate(actions) if act==0]
            # STOP repeats its input image, so the last pre-STOP occurrence is
            # expected. Count only earlier appearances as endpoint aliasing.
            end=n-int(actions[-1]==0)
            arrival=next((j for j,s in enumerate(actual[:end]) if s==actual[-1]),None)
            counts['episodes']+=1;counts['frames']+=n+1;counts['distinct_state_visits']+=len(set(actual))
            counts['aliased_transition_conflicts']+=conflicts;counts['segments_with_conflicts']+=int(conflicts>0)
            counts['endpoint_aliased_before_actual_arrival']+=int(arrival is not None)
            counts['planned_short_STOP']+=int(bool(route['actions']) and route['actions'][-1]==0 and len(route['actions'])<n)
            records.append(dict(episode=w['episode_id'],demonstration_actions=actions,actual_ids=actual,
                plan=route['actions'],subgoal=int(pg.subgoal[0]),first_endpoint_visit=arrival,conflicts=conflicts))
result=dict(training_only=True,checkpoint=a.checkpoint,update=saved['update'],counts=dict(counts),
    corpus_windows=len(windows),corpus_goal_index_histogram=dict(Counter(int(w['goal_index']) for w in windows)),
    corpus_start_histogram=dict(Counter(int(w['start']) for w in windows)),
    initial_STOP_windows=sum(int(w['actions'][0])==0 for w in windows),records=records)
Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
