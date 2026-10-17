import json,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');source=base/'symbolic_paired_v12';root=base/'symbolic_conjunctive_v23'
assert not root.exists()
for name in ['nesy_vla','scripts']:
    shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir();template=json.loads((base/'symbolic_branch_v17/configs/branch_current.json').read_text())
for name,semantics in [('branch_control','branch_tree'),('conjunctive','conjunctive_rules')]:
    cfg=json.loads(json.dumps(template));cfg.update(root=str(root),architecture='learned_conjunction_rules_v23',
        demonstrations_per_rank=48,rollouts_per_rank=3,updates=30,snapshot_every=30,counterfactual_symbol_bc=False)
    cfg['program']['action_semantics']=semantics
    cfg['program']['route_temperature']=4.
    (root/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
record={'time':time.time(),'state':'prepared_not_started','gpus_per_arm':3,'source':str(source),
    'actual_change':'Action rule conditions include generated true/false/unused choices for every learned predicate, combined by conjunction.',
    'no_new_visual_action_head':True,'no_manual_predicate_semantics':True,
    'same_conditions':['source weights','training examples','task tokens','temperature4','losses','optimizer','random seed','144 demos/update','32 actions','revision every10'],
    'fixed_validation_unchanged':True,'launch_only_after_diagnostic_review':True}
(root/'experiment.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
