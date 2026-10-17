import json
import hashlib
import shutil
from pathlib import Path
import subprocess
import time
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'run_boundary_balanced_v43';source=ROOT/'src_boundary_balanced_v43'
old=json.loads((out/'supervisor.json').read_text())
p=Path(f"/proc/{old['pid']}/cmdline")
assert not p.exists() or not p.read_bytes(),'Existing supervisor is live'
commit=json.loads((out/'latest.COMMITTED.json').read_text());assert (out/commit['filename']).exists()
assert hashlib.sha256((out/commit['filename']).read_bytes()).hexdigest()==commit['sha256']
assert shutil.disk_usage(ROOT).free>1024**3,'Insufficient free space for recovery'
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try:args=proc.read_bytes().decode(errors='replace').split('\0')
    except (OSError,PermissionError):continue
    assert str(source/'train.py') not in args,'A training rank or launcher is still live'
repair=dict(time=time.time(),checkpoint=commit,first_update=commit['update']+1,
    storage='Immutable per-rank model/code snapshots plus exact context recipes; full table digest checked on replay',
    rule_storage_roundtrip=json.loads((ROOT/'diagnostic_program_archive_v43/report.json').read_text()),
    incomplete_evaluation=dict(update=4500,status='incomplete due to disk full; preserved, not rerun or counted as complete'),
    free_bytes=shutil.disk_usage(ROOT).free)
(out/'disk_repair_resume.json').write_text(json.dumps(repair,indent=2))
with (out/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(source/'supervisor.py')],cwd=source,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,time=time.time(),resume_commit=commit)))
