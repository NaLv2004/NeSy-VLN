"""Resume v42's complete final state with boundary-balanced training samples."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=ROOT/'src_boundary_balanced_v43';out=ROOT/'run_boundary_balanced_v43';old=ROOT/'run_grounded_memory_v42'
def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return b''
supervisor=json.loads((old/'supervisor.json').read_text())
assert not command(supervisor['pid']), 'Wait for v42 supervisor exit'
for rank in (0,1):
    status=json.loads((old/f'rank{rank}.status.json').read_text())
    assert b'src_grounded_memory_v42/train.py' not in command(status['pid'])
assert not (out/'launcher.json').exists(), 'Existing run requires resume_owned.py'
contracts=json.loads((source/'sampling_contracts.json').read_text());assert contracts['passed']
assert contracts['unchanged_data_split'] and contracts['unchanged_model_and_planner']
if not (source/'predictive_nesy').exists():
    shutil.copytree(ROOT/'src_grounded_memory_v42/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
commit=json.loads((old/'latest.COMMITTED.json').read_text());checkpoint=old/commit['filename']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']
out.mkdir(exist_ok=True)
record=dict(source_checkpoint=str(checkpoint),source_commit=commit,created=time.time(),
    predecessor_best=json.loads((old/'best.COMMITTED.json').read_text()),
    restore=['model','optimizer','rank_rng_states'],gpus=[6,7],contracts=contracts,
    effect_on_navigation='Unproven; unchanged full dev80 and independent finalfixed80')
(out/'launcher.json').write_text(json.dumps(record,indent=2))
with (out/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(source/'supervisor.py')],cwd=source,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,time=time.time(),source_commit=commit,run=str(out))))
