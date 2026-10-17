"""Verify real causal replay, auxiliary gradients and unchanged action proofs."""
import json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,DataBank,atomic,initialize_execution_supervision
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstrations_many
from nesy_vla.training import train_batch
from nesy_vla.diagnostics import audit_policies
torch.set_num_threads(2);torch.manual_seed(34)
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');cfg.update(execution_outcome_supervision=True,
    execution_outcome_root='/home/user/xjy/strips_wm_navigation_20261009/data/search_cost_supervision_v3',
    recovery_demonstration_root=str(base/'data/recovery_demonstrations_v27'))
cfg['loss']['execution']=.5;setup(cfg);model=NavigationModel(cfg).eval()
if '--rule-cost' in sys.argv:
    cfg['rule_execution_cost']=True;cfg['loss']['rule_execution']=.5
initialization=initialize_execution_supervision(source,model);bank=DataBank(cfg)
selected=[next(e for e in bank.recovery_demo_ids if bank.by_id[e]['category']==category) for category in ['chair','table']]
policies=demonstrations_many(model,bank,selected);count=0
for policy,eid in zip(policies,selected):
    actual=[r for c in policy.chunks for r in c['rows']]
    expected=bank.demos[eid]
    assert len(actual)==len(expected)
    for a,e in zip(actual,expected):
        assert a.get('execution_outcome_labels')==e.get('execution_outcome_labels')
        count+=int('execution_outcome_labels' in a)
    assert all('execution_outcome_labels' not in f for f in policy.feedback)
assert count>0
proof=audit_policies(model,policies);assert proof['discrete_boundary'] and proof['replay_probability_max_error']<1e-5
policy=policies[0];row=policy.chunks[0]['rows'][0];p=policy.chunks[0]['program']
with torch.no_grad():
    before=model.step(p,row['image'],[],None)['probs'].clone()
    for parameter in model.execution_decoder.parameters():parameter.normal_(0,5.)
    after=model.step(p,row['image'],[],None)['probs']
assert torch.equal(before,after),'Auxiliary predictions leaked into actions'
# Restore a sensible decoder before checking real training gradients.
from nesy_vla.outcome_supervision import SymbolicExecutionDecoder
model.execution_decoder=SymbolicExecutionDecoder(cfg['program']['d'],cfg['program']['a'])
model.grounder.vision.requires_grad_(False)
model.train();loss,stats=train_batch(model,[(p.chunks[0],True) for p in policies]);loss.backward()
norms={prefix:sum(float(v.grad.detach().square().sum()) for n,v in model.named_parameters()
    if n.startswith(prefix) and v.grad is not None)**.5 for prefix in ['generator.','grounder.query.','logic.','execution_decoder.']}
assert all(torch.isfinite(torch.tensor(v)) and v>0 for v in norms.values())
assert sum(s.get('execution_rows_metric',0) for s in stats)>0
result={'time':time.time(),'passed':True,'integration_only':True,'original_fixed_or_dev_used':False,
    'source_commit':read(source/'COMMITTED'),'initialization':initialization,'provenance':bank.provenance,
    'episodes':selected,'execution_labeled_rows':count,'gradient_norms':norms,'symbolic_audit':proof,
    'auxiliary_parameter_randomization_action_probability_error':float((before-after).abs().max()),
    'execution_labels_absent_from_causal_policy_feedback':True,'statistics':stats}
for s in result['statistics']:s.pop('_observations',None)
result['rule_execution_cost_enabled']=cfg.get('rule_execution_cost',False)
atomic(base/('rule_execution_integration_v35.json' if '--rule-cost' in sys.argv else 'execution_integration_v34.json'),result)
print(json.dumps(result,indent=2))
