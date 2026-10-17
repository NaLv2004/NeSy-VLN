import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');folder=base/'evidence_archives'
record=folder/'active_maintenance_process.json'
if record.exists():
    old=json.loads(record.read_text());p=Path('/proc')/str(old['pid'])/'cmdline'
    assert not p.exists() or b'maintain_active_evidence_storage.py' not in p.read_bytes(),'Already running'
with (folder/'active_maintenance_launcher.log').open('ab') as log:
    child=subprocess.Popen(['python3',str(base/'maintain_active_evidence_storage.py')],cwd=base,
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
info={'pid':child.pid,'time':time.time(),'purpose':'Preserve exact trace evidence while reclaiming duplicate storage every600s; exits after active training supervisors finish.'}
record.write_text(json.dumps(info,indent=2));print(json.dumps(info))
