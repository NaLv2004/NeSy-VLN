"""Counterfactual goal-token test on paired original-training demonstrations."""
import gc,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
b=Path('/home/user/xjy/nesy_vla_20261010');source=b/'factorized_automaton_v51';out=b/'automaton_goal_use_v54';out.mkdir(exist_ok=True)
sys.path.insert(0,str(source/'src'))
from nesy_vla.runtime import read,setup,DataBank,sha,atomic
from nesy_vla.factorized_automaton import FactorizedTokenAutomaton
torch.set_num_threads(2);cfg=read(b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
from stripsnav.io import load_committed
rows=read(b/'spatial_automaton_v43/training_rows.json');codes=np.load(b/'spatial_automaton_v43/visual_tokens.npy');first={}
for i,r in enumerate(rows):first.setdefault(r['episode_id'],i)
examples=[]
for group in bank.paired_demo_groups:
 i,j=[first[eid] for eid in group];a,c=rows[i],rows[j]
 assert np.array_equal(np.asarray(Image.open(a['path'])),np.asarray(Image.open(c['path']))),'Paired starting pixels differ'
 if a['label']==c['label']:continue
 examples.append({'ids':list(group),'indices':[i,j],'labels':[a['label'],c['label']],
                  'tasks':[['chair','table'].index(a['category']),['chair','table'].index(c['category'])]})
assert examples
result={'time':time.time(),'original_training_pairs_with_different_expert_first_actions':len(examples),
 'source_rows_sha256':sha(b/'spatial_automaton_v43/training_rows.json'),'official_heldout_used':False,
 'qualification':'Same actual starting pixels and integer tokens, only requested goal changes. Action-label match is a training diagnostic, not navigation success.','arms':{}}
for arm in ['recurrent','reset_memory']:
 checkpoint=source/arm/'step_000020';saved=load_committed(checkpoint);m=FactorizedTokenAutomaton(**saved['manifest']['architecture']).cuda().eval();m.load_state_dict(saved['model'])
 records=[]
 with torch.no_grad():
  for e in examples:
   x=torch.tensor(codes[e['indices'][0]],device='cuda',dtype=torch.long)[None].repeat(2,1)
   tasks=torch.tensor(e['tasks'],device='cuda');p=m.generate(tasks,x[:,None],0);_,prior=m.initial_state(2,'cuda');r=m.step(p,x,prior,torch.full((2,),17,device='cuda',dtype=torch.long),0)
   actions=r['probabilities'].argmax(-1).tolist()
   records.append(dict(e,actions=actions,correct=[a==y for a,y in zip(actions,e['labels'])],
                       probability_total_variation=float((r['probabilities'][0]-r['probabilities'][1]).abs().sum()/2),
                       observed_ids_changed=int((r['observed_ids'][0]!=r['observed_ids'][1]).sum()),
                       state_ids_changed=int((r['state_ids'][0]!=r['state_ids'][1]).sum()),
                       program_coefficient_difference=float((p.coefficients[0]-p.coefficients[1]).norm())))
 result['arms'][arm]={'commit':read(checkpoint/'COMMITTED'),'pairs':len(records),'action_changes':sum(r['actions'][0]!=r['actions'][1] for r in records),
  'matched_demonstrated_actions':sum(sum(r['correct']) for r in records),'actions':2*len(records),
  'mean_probability_total_variation':float(np.mean([r['probability_total_variation'] for r in records])),
  'mean_observed_ids_changed':float(np.mean([r['observed_ids_changed'] for r in records])),
  'mean_state_ids_changed':float(np.mean([r['state_ids_changed'] for r in records])),'examples':records}
 del m,saved;gc.collect();torch.cuda.empty_cache()
atomic(out/'result.json',result);print(json.dumps({**result,'arms':{k:{a:b for a,b in v.items() if a!='examples'} for k,v in result['arms'].items()}},indent=2))
