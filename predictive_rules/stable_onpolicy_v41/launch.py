import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=ROOT/'src_stable_onpolicy_v41';out=ROOT/'run_stable_onpolicy_v41';old=ROOT/'run_probabilistic_rules_v40'
out.mkdir(exist_ok=True)
def live(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.exists() and bool(p.read_bytes())
status=json.loads((old/'supervisor.json').read_text())
assert not live(status['pid']),'v40 must finish its graceful drain first'
for rank in (0,1):
    rank_status=json.loads((old/f'rank{rank}.status.json').read_text())
    command=Path(f'/proc/{rank_status["pid"]}/cmdline')
    assert not command.exists() or b'src_probabilistic_rules_v40/train.py' not in command.read_bytes(), 'An old v40 rank is still active'
if (out/'supervisor.json').exists():
    assert not live(json.loads((out/'supervisor.json').read_text())['pid']),'v41 already live'
cache=ROOT/'onpolicy_successors_v40'
progress=json.loads((cache/'status.json').read_text())
assert progress['state']=='complete' and progress['completed']==120
readiness=json.loads((source/'data_readiness.json').read_text())
assert readiness['ready'] and readiness['all_hashes_verified'] and readiness['unique_new_episodes']==120
if not (source/'predictive_nesy').exists():
    shutil.copytree(ROOT/'src_probabilistic_rules_v40/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
commit=json.loads((old/'best.COMMITTED.json').read_text())
checkpoint=old/commit['filename']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']
(out/'launcher.json').write_text(json.dumps(dict(source_checkpoint=str(checkpoint),source_commit=commit,
    data_cache=str(cache),data_collection=progress,created=time.time()),indent=2))
with (out/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(source/'supervisor.py')],cwd=source,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,time=time.time(),source_checkpoint=commit,run=str(out))))
