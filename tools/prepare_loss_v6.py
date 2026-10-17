import json,os,hashlib,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');source=base/'symbolic_sft_v5';root=base/'symbolic_loss_v6'
root.mkdir(exist_ok=True)
for name in ['nesy_vla','scripts','configs','tests']:
    target=root/name
    if target.exists():raise RuntimeError('Will not overwrite an existing experiment')
    shutil.copytree(source/name,target,ignore=shutil.ignore_patterns('__pycache__'))
origin=source/'initialization_source';target=root/'initialization_source';target.mkdir()
marker=json.loads((origin/'COMMITTED').read_text());checkpoint=origin/marker['checkpoint_file']
os.link(checkpoint,target/checkpoint.name)
assert hashlib.sha256((target/checkpoint.name).read_bytes()).hexdigest()==marker['checkpoint_sha256']
(target/'COMMITTED').write_text(json.dumps(marker,indent=2))
base_config=json.loads((source/'configs/sft_plain.json').read_text())
base_config.update(root=str(root),initialize_from=str(target),demonstrations_per_rank=144,rollouts_per_rank=6)
for name,balanced in [('loss_reference',False),('loss_balanced',True)]:
    config=dict(base_config,architecture=name+'_v6',balanced_transition_loss=balanced)
    (root/'configs'/f'{name}.json').write_text(json.dumps(config,indent=2))
(root/'experiment.json').write_text(json.dumps({'time':time.time(),'source':str(origin),'initialization':marker,
    'comparison':'Matched single-GPU SFT arms; only balanced_transition_loss differs.',
    'demonstrations_per_update':144,'training_mode':'supervised','horizon':32,'revision_interval':10,
    'qualification':'Offline effect-fitting results are not navigation success; use original fixed80 unchanged.'},indent=2))
print(json.dumps({'root':str(root),'source_epoch':marker['epoch']}))
