"""Prepare an isolated code directory; never copy or change a running run."""
import json,shutil
from pathlib import Path

parent=Path('/home/user/xjy/nesy_vla_20261010')
source=parent/'v2';target=parent/'v3'
assert target.parent==parent and source.parent==parent
target.mkdir(exist_ok=True)
for name in ['nesy_vla','scripts','tools','configs','tests']:
    shutil.copytree(source/name,target/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
print(json.dumps({'prepared':str(target),'no_runs_copied':True}))
