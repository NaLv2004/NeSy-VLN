from pathlib import Path
import shutil
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=r/'src_rule_planning_v32';assert not out.exists()
shutil.copytree(r/'src_raw_grounding_v31',out,ignore=shutil.ignore_patterns('__pycache__'))
print(out)
