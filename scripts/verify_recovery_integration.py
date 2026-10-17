"""Real-data replay/gradient integration check, CPU only, no navigation score."""
import json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,load_checkpoint,DataBank,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstrations_many
from nesy_vla.training import train_batch
from nesy_vla.diagnostics import audit_policies
torch.set_num_threads(2);torch.manual_seed(27)
source=base/'symbolic_task_v15/runs/paired_tasks/latest'
cfg=read(source.parent/'config.json');cfg.update(recovery_demonstration_root=str(base/'data/recovery_demonstrations_v27'))
cfg['program'].update(action_semantics='branch_tree',route_temperature=4.)
setup(cfg);model=NavigationModel(cfg).eval();load_checkpoint(source,model)
bank=DataBank(cfg);selected=[]
for category in ['chair','table']:
    for collision in [True,False]:
        selected.append(next(e for e in bank.recovery_demo_ids if bank.by_id[e]['category']==category and
            any(r['public_feedback']['collision'] for r in bank.demos[e] if not r['supervision_mask'])==collision))
assert len(set(selected))==4
policies=demonstrations_many(model,bank,selected)
for policy,eid in zip(policies,selected):
    replay=[r for c in policy.chunks for r in c['rows']];expected=bank.demos[eid]
    assert [r['supervision_mask'] for r in replay]==[r['supervision_mask'] for r in expected]
    assert [r['action'] for r in replay]==[r['action'] for r in expected]
    assert [r['collision'] for r in replay]==[r['public_feedback']['collision'] for r in expected]
    assert replay[-1]['action']==0 and replay[-1]['supervision_mask']
proof=audit_policies(model,policies);assert proof['discrete_boundary'] and proof['replay_probability_max_error']<1e-5
# Freeze the encoder only for the CPU gradient check; trainable grounding and
# generated rule parameters retain gradients. Full experiments train the tower.
model.grounder.vision.requires_grad_(False)
model.train();loss,stats=train_batch(model,[(policy.chunks[0],True) for policy in policies]);loss.backward()
norms={prefix:sum(float(p.grad.detach().square().sum()) for n,p in model.named_parameters()
    if n.startswith(prefix) and p.grad is not None)**.5 for prefix in ['generator.','grounder.query.','logic.']}
assert all(torch.isfinite(torch.tensor(v)) and v>0 for v in norms.values())
assert sum(s['expert_context_count'] for s in stats)==sum(not r['supervision_mask'] for policy in policies for r in policy.chunks[0]['rows'])
result={'time':time.time(),'source_commit':read(source/'COMMITTED'),'integration_only':True,'fixed_data_used':False,
    'bank_provenance':bank.provenance,'episodes':selected,'loss':float(loss.detach()),'gradient_norms':norms,
    'context_actions_excluded_from_BC':sum(s['expert_context_count'] for s in stats),'symbolic_audit':proof,
    'source_action_semantics_changed_for_integration_only':'total branch rules, temperature4','passed':True}
atomic(base/'recovery_integration_v27.json',result);print(json.dumps(result,indent=2))
