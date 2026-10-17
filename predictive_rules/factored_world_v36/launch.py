import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_factored_world_v36';OUT.mkdir(exist_ok=True)
assert (ROOT/'v35_paused_for_v36.json').exists(),'Previous owned training must be drained first'
if (OUT/'supervisor.json').exists():
    status=json.loads((OUT/'supervisor.json').read_text());pid=status['pid']
    if Path(f'/proc/{pid}/cmdline').exists() and 'src_factored_world_v36/supervisor.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace'):
        raise RuntimeError('Existing supervisor is live; do not duplicate')
with (OUT/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(ROOT/'src_factored_world_v36/supervisor.py')],
        cwd=ROOT/'src_factored_world_v36',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,time=time.time(),run=str(OUT))))
