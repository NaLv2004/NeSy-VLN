import shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_rule_context_probe_v13'
assert not source.exists()
shutil.copytree(root/'src_definition_world_v12',source,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(root/'diagnostics/learned_rule_context.py',source/'predictive_nesy/learned_rule_context.py')
print(str(source))
