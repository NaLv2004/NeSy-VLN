import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_discrete_closure_v38';OUT.mkdir(exist_ok=True)
old=ROOT/'run_learned_persistence_v37'
previous=json.loads((old/'supervisor.json').read_text())
def alive(pid):
    path=Path(f'/proc/{pid}/cmdline')
    return path.exists() and bool(path.read_bytes())
assert not alive(previous['pid']),'Previous owned training must be drained first'
if not (ROOT/'src_discrete_closure_v38/predictive_nesy').exists():
    import shutil
    shutil.copytree(ROOT/'src_learned_persistence_v37/predictive_nesy',ROOT/'src_discrete_closure_v38/predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
if (OUT/'supervisor.json').exists():
    status=json.loads((OUT/'supervisor.json').read_text());pid=status['pid']
    if Path(f'/proc/{pid}/cmdline').exists() and 'src_discrete_closure_v38/supervisor.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace'):
        raise RuntimeError('Existing supervisor is live; do not duplicate')
commit=json.loads((old/'best.COMMITTED.json').read_text())
source=old/commit['filename']
(OUT/'launcher.json').write_text(json.dumps(dict(source_checkpoint=str(source),source_commit=commit,created=time.time()),indent=2))
with (OUT/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(ROOT/'src_discrete_closure_v38/supervisor.py')],
        cwd=ROOT/'src_discrete_closure_v38',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,time=time.time(),run=str(OUT))))
