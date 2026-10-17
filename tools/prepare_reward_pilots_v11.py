import hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_reward_v11'
assert not root.exists();root.mkdir()
source=base/'symbolic_joint_v9'
for name in ['nesy_vla','scripts','tests']:shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir()
origin=base/'symbolic_robust_v7/runs/robust_multieffect/best_development'
marker=json.loads((origin/'COMMITTED').read_text());target=root/'initialization_source';target.mkdir()
file=origin/marker['checkpoint_file'];os.link(file,target/file.name)
assert marker['optimizer_included'] and hashlib.sha256((target/file.name).read_bytes()).hexdigest()==marker['checkpoint_sha256']
(target/'COMMITTED').write_text(json.dumps(marker,indent=2))
cfg=json.loads((source/'configs/joint.json').read_text())
cfg.update(root=str(root),architecture='controlled_reward_v11',initialize_from=str(target),updates=30,
    training_chunk_batch=8,maximum_chunk_batch=8,training_frame_budget=64,revision_chunk_limit=8,
    candidate_every=5,snapshot_every=30,evaluation_every=10)
overrides={'false_stop':-40.,'repeat_action':-.10,'repeat_action_growth':-.025,'repeat_action_max':-.25}
for name,changed in [('original',False),('bounded_repeat_stop40',True)]:
    arm=dict(cfg)
    if changed:arm['training_reward_overrides']=overrides
    (root/'configs'/f'{name}.json').write_text(json.dumps(arm,indent=2))
    wrapper=root/f'run_{name}.py'
    wrapper.write_text("import runpy,sys,torch\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root))\ntorch.cuda.set_per_process_memory_fraction(.30)\nsys.argv=['scripts/train.py','--config','configs/"+name+".json','--out',str(root/'runs/"+name+"')]\nrunpy.run_path(str(root/'scripts/train.py'),run_name='__main__')\n")
tiny=dict(cfg,updates=1,rollouts_per_rank=1,demonstrations_per_rank=1,training_chunk_batch=2,
    maximum_chunk_batch=2,training_frame_budget=20,revision_chunk_limit=2,candidate_every=1,
    training_reward_overrides=overrides)
(root/'configs/integration.json').write_text(json.dumps(tiny,indent=2))
wrapper=root/'integration.py'
wrapper.write_text("import runpy,sys,torch\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root))\ntorch.cuda.set_per_process_memory_fraction(.30)\nsys.argv=['scripts/train.py','--config','configs/integration.json','--out',str(root/'integration')]\nrunpy.run_path(str(root/'scripts/train.py'),run_name='__main__')\n")
manifest={'time':time.time(),'source_checkpoint':marker,'source':str(origin),
    'source_selected_by':'Best development80 score, not fixed80','source_score':json.loads((origin.parent/'best_score.json').read_text()),
    'controlled_difference':overrides,'evidence':str(base/'reward_diagnostic_v11/counterfactual.json'),
    'matched':['checkpoint with optimizer','random seeds','data splits','network architecture','8 rollouts and12 demos per update','30 updates'],
    'unchanged':['evaluation success definition','all actions through generated rules','learned predicates only','discrete inference inputs','32-action horizon','revision every10 actions'],
    'all_recurrences_penalized':True,'navigation_improvement_unproven':True}
(root/'experiment.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'root':str(root),'source':marker}))
