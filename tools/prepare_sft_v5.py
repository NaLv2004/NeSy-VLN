"""Pin the exact already committed small-model weights without disturbing training."""
import json,os,hashlib,shutil,time
from pathlib import Path
parent=Path('/home/user/xjy/nesy_vla_20261010');source=parent/'discrete_v4';root=parent/'symbolic_sft_v5'
root.mkdir(exist_ok=True)
for name in ['nesy_vla','scripts','configs','tests']:
    shutil.copytree(source/name,root/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
origin=source/'runs/discrete_gru/latest';target=root/'initialization_source';target.mkdir(exist_ok=True)
if not (target/'COMMITTED').exists():
    for attempt in range(3):
        marker=json.loads((origin/'COMMITTED').read_text());file=origin/marker['checkpoint_file']
        try:os.link(file,target/file.name)
        except FileNotFoundError:continue
        except FileExistsError:pass
        h=hashlib.sha256()
        with (target/file.name).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        assert h.hexdigest()==marker['checkpoint_sha256']
        (target/'COMMITTED').write_text(json.dumps(marker,indent=2));break
    else:raise RuntimeError('Could not pin a committed initialization checkpoint')
marker=json.loads((target/'COMMITTED').read_text())
base=json.loads((source/'configs/discrete_gru.json').read_text())
base.update(root=str(root),initialize_from=str(target),training_mode='supervised',updates=200,
    demonstrations_per_rank=48,training_chunk_batch=32,maximum_chunk_batch=64,
    training_frame_budget=256,revision_chunk_limit=32,learning_rate=.0003,vision_learning_rate=.00003)
base['loss']['sparse']=0.
for name,spatial in [('sft_plain',False),('sft_spatial',True)]:
    cfg=dict(base,architecture=name+'_v5',spatial_encoding=spatial)
    (root/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
(root/'experiment.json').write_text(json.dumps({'time':time.time(),'source':str(origin),'pinned':marker,
    'comparison':'Only image positional encoding differs between the two SFT arms.',
    'shared_changes':['supervised demonstration training through full symbolic program','complete shuffled epochs',
        'higher learning rates','initial sparsity regularizer disabled','larger useful frame batches'],
    'splits':'unchanged and asserted on every startup','validation':'existing development80 and fixed80 every 10 updates',
    'success_not_yet_proven':True},indent=2))
print(json.dumps({'root':str(root),'initialization':marker,'arms':['sft_plain','sft_spatial']},indent=2))
