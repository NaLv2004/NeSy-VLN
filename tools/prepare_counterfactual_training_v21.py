"""Prepare matched full-trajectory controls. Launch is a separate decision."""
import json,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');source=base/'symbolic_paired_v12';root=base/'symbolic_counterfactual_v21'
assert not root.exists()
for name in ['nesy_vla','scripts']:
    shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir();original=json.loads((base/'symbolic_branch_v17/configs/branch_current.json').read_text())
configs={}
for name,enabled in [('straight_through',False),('counterfactual',True)]:
    cfg=json.loads(json.dumps(original))
    cfg.update(root=str(root),architecture='generated_rules_discrete_gradient_comparison_v21',
        demonstrations_per_rank=48,rollouts_per_rank=3,updates=30,snapshot_every=30,
        counterfactual_symbol_bc=enabled)
    cfg['program']['route_temperature']=4.
    (root/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2));configs[name]=cfg
differing=[key for key in configs['counterfactual'] if configs['counterfactual'][key]!=configs['straight_through'][key]]
assert differing==['counterfactual_symbol_bc']
record={'time':time.time(),'state':'prepared_not_started','source':str(source),'gpus_per_arm':3,
    'difference':differing,'unchanged':['initial weights','training demonstrations and paired goals','random seed',
        'temperature4 generated action conclusions','learning rates','all auxiliary losses','32-action horizon',
        '10-action program revisions','isolated development80 and fixed80'],
    'test':'Compare ordinary local surrogate gradients with exact one-bit symbolic-loss differences.',
    'no_navigation_success_claim':True}
(root/'experiment.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
