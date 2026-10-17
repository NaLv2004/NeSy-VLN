from pathlib import Path
import shutil

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
destination=root/'src_adaptive_observed_v26'
assert not destination.exists()
destination.mkdir()
shutil.copytree(root/'src_state512_v25/predictive_nesy',destination/'predictive_nesy')
shutil.copy2(root/'src_state512_v25/evaluate_state512_v25.py',destination/'evaluate_state512_v25.py')
print(destination)
