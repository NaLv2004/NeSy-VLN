import shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_discrete_visual_world_v15';assert not source.exists()
shutil.copytree(root/'src_definition_world_v12',source,ignore=shutil.ignore_patterns('__pycache__'))
for name,target in [('discrete_visual_world.py','predictive_nesy/discrete_visual_world.py'),
    ('build_discrete_visual_world_data.py','build_discrete_visual_world_data.py'),('train_discrete_visual_world_v15.py','train_discrete_visual_world_v15.py')]:
    shutil.copy2(root/'diagnostics'/name,source/target)
print(str(source))
