"""Measure numerical batch sensitivity using saved observations; no new episodes."""
import json,os,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'numerical_symbol_batch_v63';out.mkdir(exist_ok=True)
assert not (out/'result.json').exists()
sys.path.insert(0,str(b/'probabilistic_automaton_v56/src'))
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
from nesy_vla.runtime import read,atomic,setup
setup(read(b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json'))
from stripsnav.io import load_committed
run=b/'probabilistic_automaton_v56/soft_recurrent';loc=run/'step_000040';payload=load_committed(loc)
m=ProbabilisticTokenAutomaton(**payload['manifest']['architecture']).cuda().eval();m.load_state_dict(payload['model'])
traces=[read(p) for p in sorted((run/'evaluations/step_000040').glob('*/trace.json'))]
assert len(traces)==80
x=torch.tensor([t['steps'][0]['patch_tokens'] for t in traces],device='cuda')
tasks=torch.tensor([['chair','table'].index(t['episode']['category']) for t in traces],device='cuda')
c=torch.tensor([t['programs'][0]['coefficients'] for t in traces],device='cuda')
expected=torch.tensor([t['steps'][0]['probabilities'] for t in traces],device='cuda')
observed=torch.tensor([t['steps'][0]['observation_tokens'] for t in traces],device='cuda')
result={'checkpoint':read(loc/'COMMITTED'),'scope':'Same eighty saved initial observations and stored program coefficients, no simulator, no new navigation evaluations.','cases':[]}
with torch.no_grad():
 for tf32 in [True,False]:
  outputs={}
  torch.backends.cuda.matmul.allow_tf32=tf32
  for batch in [1,8,16,80]:
   probabilities=[];ids=[];margins=[]
   for offset in range(0,80,batch):
    xx=x[offset:offset+batch];tt=tasks[offset:offset+batch]
    p=m.program_from_coefficients(c[offset:offset+batch],0,tt)
    _,memory=m.initial_state(len(xx),'cuda');a=torch.full((len(xx),),17,device='cuda',dtype=torch.long)
    r=m.step(p,xx,memory,a,0);probabilities.append(r['probabilities']);ids.append(r['observed_ids'])
    values=r['observed_probabilities'].topk(2,-1).values;margins.append(values[...,0]-values[...,1])
   outputs[batch]=(torch.cat(probabilities),torch.cat(ids),torch.cat(margins))
  for batch,(probs,oid,margin) in outputs.items():
   reference=outputs[8]
   result['cases'].append({'tf32':tf32,'batch':batch,
    'max_probability_error_vs_recorded':float((probs-expected).abs().max()),
    'changed_actions_vs_recorded':int((probs.argmax(-1)!=expected.argmax(-1)).sum()),
    'changed_observation_ids_vs_recorded':int((oid!=observed).sum()),
    'max_probability_error_vs_same_mode_batch8':float((probs-reference[0]).abs().max()),
    'changed_observation_ids_vs_same_mode_batch8':int((oid!=reference[1]).sum()),
    'minimum_observation_winner_margin':float(margin.min())})
result['finished']=time.time();atomic(out/'result.json',result);print(json.dumps(result,indent=2))
