from pathlib import Path
import shutil

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
destination=root/'src_open_loop_v34'
assert not destination.exists()
shutil.copytree(root/'src_action_grounded_v33',destination,ignore=shutil.ignore_patterns('__pycache__'))
print(destination)
