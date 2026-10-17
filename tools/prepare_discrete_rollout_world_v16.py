"""Snapshot the independent multi-step component; no live source is modified."""
import json,shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_discrete_visual_world_v15';destination=root/'src_discrete_rollout_world_v16'
assert not destination.exists()
shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['predictive_nesy/discrete_rollout_world.py','train_discrete_rollout_world_v16.py']:
    shutil.copy2(root/'stage_discrete_rollout_v16'/name,destination/name)
print(json.dumps(dict(source=str(destination),preserved_source=str(source))))
