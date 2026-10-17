"""Intervene on learned observation IDs, with all other saved premises fixed."""
import os,json,sys,time,hashlib
from pathlib import Path
import numpy as np,torch
B=Path('/home/user/xjy/nesy_vla_20261010');RUN=B/'parallel_ppo_v85';OUT=B/'state_semantics_v91'
sys.path.insert(0,str(RUN/'src'))
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
read=lambda p:json.loads(p.read_text())
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=True
checkpoint=read(RUN/'best_development/COMMITTED');file=RUN/'best_development'/checkpoint['checkpoint_file']
assert checkpoint['epoch']==70 and hashlib.sha256(file.read_bytes()).hexdigest()==checkpoint['checkpoint_sha256']
saved=torch.load(file,map_location='cpu',weights_only=False)
model=ProbabilisticTokenAutomaton(**saved['manifest']['architecture']).cuda().eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
del saved
trials=[read(p) for p in sorted((RUN/'evaluations/step_000070').glob('*/trace.json'))]
used={(task,f):sorted({s['observation_tokens'][f] for t in trials if t['episode']['category']==task for s in t['steps']}) for task in ['chair','table'] for f in range(4)}
records=[];max_error=0.;programs=[]
with torch.no_grad():
 for t in trials:
  e=t['episode'];task=['chair','table'].index(e['category']);pg={p['start']:p for p in t['programs']}
  seen=set()
  for s in t['steps']:
   if s['image_sha256'] in seen:continue
   seen.add(s['image_sha256'])
   c=torch.tensor(pg[s['program_start']]['coefficients'],device='cuda').reshape(1,-1).repeat(8,1)
   program=model.program_from_coefficients(c,s['program_start'],torch.tensor([task]*8,device='cuda'))
   o=torch.tensor(s['observation_tokens'],device='cuda')[None].repeat(8,1)
   prev=torch.tensor(s['previous_belief'],device='cuda')[None].repeat(8,1,1);a=torch.tensor([s['previous_action']]*8,device='cuda')
   observed=torch.nn.functional.one_hot(o,16).float()
   belief=model.transition_belief(program,observed,prev,a,s['step'])
   base,_=model.apply_action_rules(program,belief,s['step']);expected=torch.tensor(s['probabilities'],device='cuda')
   error=float((base[0]-expected).abs().max());max_error=max(max_error,error);assert error<2e-5,error
   effects=[]
   for f in range(4):
    for symbol in used[e['category'],f]:
     modified=o.clone();modified[:,f]=symbol
     state=model.transition_belief(program,torch.nn.functional.one_hot(modified,16).float(),prev,a,s['step'])
     probability,_=model.apply_action_rules(program,state,s['step'])
     effects.append(dict(factor=f,symbol=symbol,original_symbol=int(o[0,f]),
      total_variation=float((probability[0]-base[0]).abs().sum()/2),changed_action=bool(probability[0].argmax()!=base[0].argmax()),
      stop_change=float(probability[0,0]-base[0,0]),action=int(probability[0].argmax()),probabilities=probability[0].cpu().tolist()))
   records.append(dict(episode_id=e['episode_id'],category=e['category'],scene=e['scene'],step=s['step'],effects=effects))
  for p in pg.values():programs.append(dict(episode_id=e['episode_id'],category=e['category'],start=p['start'],coefficients=p['coefficients']))
result=dict(checkpoint=checkpoint,max_saved_probability_error=max_error,records=records,programs=programs,
 scope='Exact symbolic counterfactual: change one observed categorical ID, keep generated program, previous action and prior belief fixed; no environment actions or optimization. Only IDs observed within the same task are substituted. Off-manifold combinations remain possible.')
(OUT/'interventions.json').write_text(json.dumps(result))
print(json.dumps(dict(state='complete',rows=len(records),max_error=max_error,time=time.time())))
