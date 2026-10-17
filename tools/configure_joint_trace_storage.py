import json,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_joint_v9')
assert not (root/'runs/joint/supervisor.json').exists()
path=root/'configs/joint.json';cfg=json.loads(path.read_text());cfg['training_trace_episodes_per_rank']=1
path.write_text(json.dumps(cfg,indent=2))
manifest=json.loads((root/'experiment.json').read_text())
manifest['trace_storage']={'time':time.time(),'training':'One rotating full trajectory per rank/update; all episode IDs, successes and losses retained in rank logs.',
    'evaluation':'Every episode has full trajectory, symbols, rules, exact input images and replay video.'}
(root/'experiment.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest['trace_storage']))
