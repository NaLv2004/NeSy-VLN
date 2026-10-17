from pathlib import Path
import shutil
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'src_adaptive_observed_v27';assert not out.exists();out.mkdir()
shutil.copytree(root/'src_state512_v25/predictive_nesy',out/'predictive_nesy')
shutil.copy2(root/'src_state512_v25/evaluate_state512_v25.py',out/'evaluate_state512_v25.py')
print(out)
