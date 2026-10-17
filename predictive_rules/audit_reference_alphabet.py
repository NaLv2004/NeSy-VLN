"""Read-only diagnostic of the learned reference alphabet; no reference actor."""
import json, random
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from predictive_nesy.environment import load_previous
from predictive_nesy.subgoal_data import load_windows,batch

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
torch.set_num_threads(2)
reference,manifest,commit=load_previous()
saved=torch.load(root/'run_visual_graph_subgoal_v5/checkpoint_004000.pt',map_location='cpu',weights_only=False)
windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
counts=Counter();records=[]
with torch.no_grad():
    for offset in range(0,len(selected),16):
        subset=selected[offset:offset+16];d=batch(subset,'cuda')
        # Generate once at the initial boundary. Every future symbol is a label
        # computed with the same frozen learned definitions, without action tags.
        program=reference.generate(d['task'],d['history'][:,:1],0)
        features=reference.observation(reference.embed(d['targets'])).reshape(len(subset),11,reference.factors,reference.width)
        logits=torch.einsum('btfd,bfkd->btfk',F.normalize(features,dim=-1),F.normalize(program.definitions,dim=-1))
        labels=logits.argmax(-1).cpu().tolist()
        for w,values in zip(subset,labels):
            n=int(w['goal_index']);states=[tuple(v) for v in values[:n+1]];actions=w['actions'][:n].tolist()
            transition={};conflicts=0
            for s,action,t in zip(states,actions,states[1:]):
                key=(s,int(action));conflicts+=int(key in transition and transition[key]!=t);transition[key]=t
            arrival=next((i for i,s in enumerate(states[:-1]) if s==states[-1]),None)
            counts['episodes']+=1;counts['frames']+=n+1;counts['distinct_state_visits']+=len(set(states))
            counts['aliased_transition_conflicts']+=conflicts;counts['segments_with_conflicts']+=int(conflicts>0)
            counts['endpoint_already_encoded_before_demonstrated_end']+=int(arrival is not None)
            records.append(dict(episode=w['episode_id'],states=states,conflicts=conflicts,first_endpoint_visit=arrival))
result=dict(training_only=True,reference_commit=commit,factors=reference.factors,states_per_factor=reference.states,
            counts=dict(counts),reference_actor_called=False,reference_recurrent_state_correction_called=False,records=records)
(root/'diagnostics/aliasing_reference_alphabet.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
