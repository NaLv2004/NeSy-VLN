import json
from pathlib import Path
import subprocess
import time
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'run_probabilistic_rules_v40';source=ROOT/'src_probabilistic_rules_v40'
old=json.loads((out/'supervisor.json').read_text())
p=Path(f"/proc/{old['pid']}/cmdline")
assert not p.exists() or not p.read_bytes(),'Existing supervisor is live'
commit=json.loads((out/'latest.COMMITTED.json').read_text());assert (out/commit['filename']).exists()
with (out/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(source/'supervisor.py')],cwd=source,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,time=time.time(),resume_commit=commit)))
