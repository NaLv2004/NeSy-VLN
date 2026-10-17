import shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_replay_boundary_v20'
assert not source.exists()
shutil.copytree(root/'src_raw_boundary_v19',source)
print(source)
