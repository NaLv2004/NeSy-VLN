from pathlib import Path
import shutil
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
target=root/'src_program_ppo_v35'
assert not target.exists()
shutil.copytree(root/'src_action_grounded_v33',target,ignore=shutil.ignore_patterns('__pycache__'))
print(target)
