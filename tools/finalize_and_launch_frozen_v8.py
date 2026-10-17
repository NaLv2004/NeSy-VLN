import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_vision_v8';source=base/'frozen_vision_v8_code'
record=json.loads((root/'controller_held.json').read_text())
assert not Path(f"/proc/{record['pid']}/cmdline").exists()
assert json.loads((root/'allocation.json').read_text())['jobs']=={}
assert not list((root/'runs').glob('*/supervisor.json'))
assert '50 passed' in (source/'verification.log').read_text()
for name in ['nesy_vla','tests']:
    shutil.copytree(source/name,root/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(source/'verification.log',root/'verification.log')
manifest=json.loads((root/'experiment.json').read_text())
manifest.update(tests_passed=50,inverse_objective='Compare hard next-state predicate codes for all candidate actions; STE gradients during training. Identical codes cannot discriminate actions using fractional confidence.',
    gate_optimization='Batch equivalent symbolic rule gates; numerical and gradient equivalence tested.')
(root/'experiment.json').write_text(json.dumps(manifest,indent=2))
history=root/'controller_history';history.mkdir(exist_ok=True)
os.rename(root/'controller.json',history/f"held_{record['pid']}.json")
exec((root/'launch_prepared.py').read_text())
