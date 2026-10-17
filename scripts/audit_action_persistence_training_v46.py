"""Compare empirical action persistence in training to completed development traces."""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
base=Path('/home/user/xjy/nesy_vla_20261010/spatial_automaton_v43');read=lambda p:json.loads(p.read_text())
rows=read(base/'training_rows.json');codes=np.load(base/'visual_tokens.npy');pairs=[]
for i in range(1,len(rows)):
 if rows[i]['episode_id']==rows[i-1]['episode_id'] and rows[i]['supervised']:
  pairs.append({'category':rows[i]['category'],'action':rows[i]['label'],'previous_action':rows[i-1]['label'],'same_tokens':bool(np.array_equal(codes[i],codes[i-1])),'repeated_action':rows[i]['label']==rows[i-1]['label']})
summary={}
for name,subset in [('all',pairs)]+[(c,[p for p in pairs if p['category']==c]) for c in ['chair','table']]:
 same=[r for r in subset if r['same_tokens']];summary[name]={'pairs':len(subset),'repeated_action':sum(r['repeated_action'] for r in subset),'identical_current_visual_tokens':len(same),'repeat_given_identical_codes':sum(r['repeated_action'] for r in same),'labels':dict(Counter(r['action'] for r in subset))}
traces=[]
for p in sorted((base/'development').glob('*/trace.json')):
 r=read(p);traces.extend((p.parent.name,s) for s in r['steps'])
# Audit whether a source-current-token substitution also changes episode.
count=sum(traces[i][0]!=traces[i-1][0] for i in range(len(traces)))
result={'time':time.time(),'training_pairs':summary,'scope':'Original training supervision only; no added labels or fitted policy. Identical token codes do not prove identical physical states.','prior_ablation_qualification':'The earlier one-row rotation mostly substituted an adjacent frame in the same episode; its action-change count cannot establish that visual evidence matters less than the previous-action token. The new controlled training ablation is needed.','globally_adjacent_other_episode_count':count,'development_actions':len(traces)}
(base/'action_persistence_training_audit_v46.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
