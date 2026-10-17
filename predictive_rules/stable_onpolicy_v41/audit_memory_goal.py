"""Frozen v40 training-cache diagnostic; never alters a policy or goal."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import ROOT,sha
from model import FactoredWorld,Config

torch.set_num_threads(3)
cache=ROOT/'onpolicy_successors_v40'
commit=json.loads((cache/'policy.COMMITTED.json').read_text())
assert sha(cache/'policy.pt')==commit['sha256']
saved=torch.load(cache/'policy.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config'])).eval()
model.load_state_dict(saved['model'],strict=True);c=model.cfg
records=[];examples=[];invariant=[];terminal_variants={}
with torch.inference_mode():
    for path in sorted((cache/'counterfactual_onpolicy').glob('*.json')):
        record=json.loads(path.read_text());assert record['training_only']
        with np.load(path.with_suffix('.npz')) as archive:
            data={key:archive[key] for key in ('features','successor_features')}
            for step in record['steps']:
                m,o=step['memory'],step['observed'];goal=step['proof']['goal'];gm,go=divmod(goal,c.observations)
                memory=F.one_hot(torch.tensor([m]),c.memories).float()
                obs=F.one_hot(torch.tensor([o]),c.observations).float()
                stop_m=int(model.memory_step(memory,obs,torch.tensor([0]))[0].argmax(-1))
                same_observation=o==go;memory_blocked=same_observation and stop_m!=gm
                row=dict(episode=record['episode'],step=step['step'],goal_memory=gm,goal_observation=go,
                    observed=o,memory=m,stop_next_memory=stop_m,chosen_action=step['action'],
                    observation_matches_goal=same_observation,blocked_only_by_goal_memory=memory_blocked)
                records.append(row)
                if memory_blocked and len(examples)<16:examples.append(row)
                if go not in terminal_variants:
                    ms=F.one_hot(torch.arange(c.memories),c.memories).float()
                    os=F.one_hot(torch.full((c.memories,),go),c.observations).float()
                    terminal=model.memory_step(ms,os,torch.zeros(c.memories,dtype=torch.long))[0].argmax(-1)
                    terminal_variants[go]=dict(distinct_post_stop_memory_ids=int(terminal.unique().numel()),
                        mapping=terminal.tolist())
                t=step['step']
                different=(data['successor_features'][t].astype(np.float32)-data['features'][t].astype(np.float32)).reshape(17,-1)
                same_raw=np.flatnonzero(np.max(np.abs(different),axis=-1)==0)
                for action in same_raw:
                    if action==0:continue
                    following=int(model.memory_step(memory,obs,torch.tensor([int(action)]))[0].argmax(-1))
                    invariant.append(dict(changed=following!=m,action=int(action),before=m,after=following))
counts=Counter(r['action'] for r in invariant)
blocked=[r for r in records if r['blocked_only_by_goal_memory']]
matched=[r for r in records if r['observation_matches_goal']]
result=dict(training_only=True,checkpoint=commit,episodes=120,executed_states=len(records),
    observation_matches_generated_goal=len(matched),goal_observation_match_fraction=len(matched)/len(records),
    matching_observation_blocked_by_goal_memory=len(blocked),
    conditional_memory_block_fraction=len(blocked)/max(1,len(matched)),
    affected_episodes=len({r['episode'] for r in blocked}),examples=examples,
    exact_unchanged_visual_effects=dict(nonstop_examples=len(invariant),
        changed_memory_fraction=sum(r['changed'] for r in invariant)/max(1,len(invariant)),by_action=dict(counts)),
    distinct_post_stop_memory_ids_by_goal_observation=terminal_variants,
    qualification='Matching an anonymous observation ID is NOT proof of physical task success. Unchanged visual evidence can legitimately update history memory. These measurements diagnose additional history constraints, not an alternative success criterion or permission to drop the generated goal.')
Path(__file__).with_name('memory_goal_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['examples','distinct_post_stop_memory_ids_by_goal_observation']}))
