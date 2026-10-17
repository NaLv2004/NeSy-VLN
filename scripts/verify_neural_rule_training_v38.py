"""Verify full causal trajectories, terminal actions and rule-only gradients."""
import json,os,sys,time
from pathlib import Path
import torch
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,DataBank,atomic,initialize_execution_supervision,load_checkpoint
from nesy_vla.model import NavigationModel
from nesy_vla.policy import demonstrations_many
from nesy_vla.training import train_batch
from nesy_vla.diagnostics import audit_policies

torch.set_num_threads(2);torch.manual_seed(3800)
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');cfg.update(execution_outcome_supervision=True,
    execution_outcome_root='/home/user/xjy/strips_wm_navigation_20261009/data/search_cost_supervision_v3',
    recovery_demonstration_root=str(base/'data/recovery_demonstrations_v27'),install_neural_conjunction_gates=True)
cfg['program']['neural_conjunction_gates']=True;cfg['loss']['execution']=.5
cfg['execution_positive_weight']=read(base/'execution_learning_v34/dataset.json')['positive_weights']
setup(cfg);model=NavigationModel(cfg).eval().to('cuda')
initialization=initialize_execution_supervision(source,model);bank=DataBank(cfg)
selected=[next(e for e in bank.recovery_demo_ids if bank.by_id[e]['category']==category) for category in ['chair','table']]
policies=demonstrations_many(model,bank,selected)
assert all(p.chunks[-1]['rows'][-1]['action']==0 for p in policies)
assert all('execution_outcome_labels' not in f for p in policies for f in p.feedback)
audit=audit_policies(model,policies);assert audit['discrete_boundary'] and audit['replay_probability_max_error']<1e-5
model.train();model.grounder.vision.requires_grad_(False)
optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4)
optimizer.zero_grad();statistics=[]
for policy in policies:
    for chunk in policy.chunks:
        loss,stats=train_batch(model,[(chunk,True)]);loss.backward();statistics.extend(stats)
norms={prefix:sum(float(v.grad.detach().square().sum()) for n,v in model.named_parameters()
    if n.startswith(prefix) and v.grad is not None)**.5 for prefix in
    ['generator.','grounder.query.','logic.action_rule_gate.','execution_decoder.']}
assert all(torch.isfinite(torch.tensor(v)) and v>0 for v in norms.values())
torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();model.eval()
chunk=policies[0].chunks[0];row=chunk['rows'][0]
with torch.no_grad():
    original=model.step(chunk['program'],row['image'],[],None)['probs'].clone()
    for parameter in model.execution_decoder.parameters():parameter.normal_(0,4.)
    randomized=model.step(chunk['program'],row['image'],[],None)['probs']
assert torch.equal(original,randomized)
for stats in statistics:stats.pop('_observations',None)
result={'passed':True,'time':time.time(),'pid':os.getpid(),'original_fixed_or_dev_used':False,
    'episodes':selected,'initialization':initialization,'proof_audit':audit,'gradient_norms':norms,
    'successful_terminal_STOP_included':True,'execution_labels_absent_from_policy_context':True,
    'auxiliary_head_randomization_probability_error':float((original-randomized).abs().max()),
    'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30,'statistics':statistics}
atomic(base/'neural_rule_training_integration_v38.json',result);print(json.dumps(result,indent=2))
