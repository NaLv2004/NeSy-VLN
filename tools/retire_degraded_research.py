"""Retire diagnosed runs at their intact checkpoint; leave reward pilots alone."""
import hashlib,json,os,signal,select,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
names=['symbolic_robust_v7/runs/robust_multieffect','symbolic_joint_v9/runs/joint',
       'symbolic_vision_v8/runs/frozen_convnext','symbolic_vision_v8/runs/frozen_qwen']
records=[];fds=[]
for name in names:
    run=(base/name).resolve();assert run.is_relative_to(base)
    commit=json.loads((run/'latest/COMMITTED').read_text());path=run/'latest'/commit['checkpoint_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==commit['checkpoint_sha256']
    supervisor=json.loads((run/'supervisor.json').read_text());pid=supervisor.get('pid')
    proc=Path(f'/proc/{pid}/cmdline');command=proc.read_bytes() if proc.exists() else b''
    if command:
        assert str(run).encode() in command and b'torch.distributed.run' in command
        assert os.getpgid(pid)==pid
        fds.append(os.pidfd_open(pid))
    record={'run':str(run),'time':time.time(),'commit':commit,'pid':pid,
            'reason':'Development decline and paired-goal conditioning failure; preserve weights for new architecture experiments.'}
    (run/'STOP').write_text(record['reason']+'\n')
    (run/'RETIRED_RESEARCH.json').write_text(json.dumps(record,indent=2));records.append(record)
    if command:os.killpg(pid,signal.SIGTERM)
deadline=time.time()+45
while fds and time.time()<deadline:
    ready=select.select(fds,[],[],max(0,deadline-time.time()))[0]
    for fd in ready:os.close(fd);fds.remove(fd)
for fd in fds:os.close(fd)
for row in records:
    p=Path(f"/proc/{row['pid']}/cmdline");row['still_live']=bool(p.exists() and p.read_bytes())
    run=Path(row['run']);current=json.loads((run/'latest/COMMITTED').read_text())
    row['latest_commit_preserved']=current==row['commit']
(base/'degraded_runs_retirement.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
