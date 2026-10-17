"""Transfer completed-run storage maintenance to the newly verified training pair."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');folder=base/'evidence_archives'
lock=(folder/'active_maintenance.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
startup=json.loads((base/'symbolic_execution_v38/startup_inspection.json').read_text())
assert all(j['commit_verified'] and j['live'] for j in startup['jobs'].values())
archive=base/'archive_active_evidence.py';source=archive.read_text()
old="'symbolic_recovery_v28/runs/base_control','symbolic_recovery_v28/runs/recovery_mixture']"
new="'symbolic_recovery_v28/runs/base_control','symbolic_recovery_v28/runs/recovery_mixture',\n    'symbolic_execution_v38/runs/detached_outcome_control','symbolic_execution_v38/runs/grounded_outcome_feedback']"
if old in source:archive.write_text(source.replace(old,new))
else:assert new in source
original=base/'maintain_active_evidence_storage.py'
if not original.exists():original=base/'tools/maintain_active_evidence_storage.py'
assert original.exists()
source=original.read_text().replace("base/'symbolic_recovery_v28'","base/'symbolic_execution_v38'")
source=source.replace("['symbolic_teacher_v25','symbolic_recovery_v28']","['symbolic_execution_v38']")
source=source.replace("active_maintenance.json","active_maintenance_v38.json").replace("active_maintenance.log","active_maintenance_v38.log")
target=base/'maintain_execution_evidence_v38.py';target.write_text(source)
meta=folder/'active_maintenance_v38_process.json'
if meta.exists():
    previous=json.loads(meta.read_text());proc=Path(f'/proc/{previous["pid"]}/cmdline')
    assert not proc.exists(),'Existing maintenance process must not be duplicated'
    assert "has no attribute 'pidfd_open'" in (folder/'maintenance_launcher_v38.log').read_text()
    (folder/'maintenance_v38_failed_environment.json').write_text(json.dumps(previous,indent=2))
assert hasattr(os,'pidfd_open'),'Use the system Python with pidfd support'
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (folder/'maintenance_launcher_v38.log').open('ab') as log:
    child=subprocess.Popen([sys.executable,'-u',str(target)],
        stdout=log,stderr=log,start_new_session=True)
record={'pid':child.pid,'time':time.time(),'script':str(target),'interval_seconds':600}
meta.write_text(json.dumps(record,indent=2));print(json.dumps(record))
