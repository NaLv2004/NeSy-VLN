"""Check whether new anonymous memory states have escaped duplicated symmetry."""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
torch.set_num_threads(1)
B=Path('/home/user/xjy/nesy_vla_20261010/consequence_refinement_v93')
p=B/'checkpoint_0400.pt';saved=torch.load(p,map_location='cpu',weights_only=False)
s=saved['model'];arch=saved['manifest']['architecture'];assert arch['states']==32 and arch['task_banks']
def difference(name,axis):
 x=s[name];a=x.index_select(axis,torch.arange(0,32,2));b=x.index_select(axis,torch.arange(1,32,2))
 d=(a-b).abs()
 return dict(max=float(d.max()),mean=float(d.mean()),fraction_exactly_equal=float((d==0).float().mean()))
result=dict(time=time.time(),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),update=saved['update'],
 axes={
  'observation_definitions':difference('definitions',2),
  'transition_observed_premise':difference('transition',2),
  'transition_previous_memory_premise':difference('transition',3),
  'transition_next_memory_conclusion':difference('transition',5),
  'generated_transition_observed':difference('transition_basis',2),
  'generated_transition_next_memory':difference('transition_basis',3),
  'action_first_memory_premise':difference('conclusions',2),
  'action_second_memory_premise':difference('conclusions',3),
  'generated_action_first_memory_premise':difference('conclusion_basis',2),
  'generated_action_second_memory_premise':difference('conclusion_basis',3)},
 scope='Read-only completed checkpoint tensor audit. No optimization or navigation evaluation. Child indices identify arbitrary initialization copies, not human semantic labels.')
(B/'symmetry_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
