"""Prepare a full generated-branch-rule experiment; no process is started here."""
import hashlib,json,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');source=base/'symbolic_paired_v12';root=base/'symbolic_branch_v17'
assert not root.exists()
for name in ['nesy_vla','scripts']:
    shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir();cfg=json.loads((base/'symbolic_task_v15/configs/paired_tasks.json').read_text())
cfg.update(root=str(root),architecture='automatically_generated_total_branch_rules_v17',
           demonstrations_per_rank=72,rollouts_per_rank=3,updates=30,snapshot_every=30)
cfg['program']['action_semantics']='branch_tree'
(root/'configs/branch_current.json').write_text(json.dumps(cfg,indent=2))
visual=json.loads(json.dumps(cfg));visual['initialize_visual_from']='/home/user/xjy/strips_wm_navigation_20261009/runs/search_cost_ppo_belief_v2_step000200/best'
(root/'configs/branch_visual.json').write_text(json.dumps(visual,indent=2))
record={'time':time.time(),'state':'prepared_not_started','source':str(source),'gpus':[6,7],
        'select_variant_only_after_capacity_probe':True,'default_variant':'branch_current',
        'new_inference':'Generated complementary predicate tests; each complete branch path concludes generated action probabilities.',
        'coverage':'For every Boolean input state, branch-path mass sums to1; no non-symbolic fallback.',
        'unchanged':['Automatically learned predicate definitions','Hard symbol boundary','Generated add/delete effects',
                     'Symbolic inverse prediction','Revision every10 actions','Maximum32 actions','Frozen train/development/fixed80 partitions'],
        'initialization':'Warm start program/grounder weights from development-selected source; fresh optimizer. Interpreter semantics deliberately changed.',
        'no_navigation_success_claim':True}
(root/'experiment.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
