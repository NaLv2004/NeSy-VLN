"""Retire only this experiment's obsolete trainer; preserve committed files."""
import hashlib,json,os,signal,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
run=root/'run_v1';record=json.loads((run/'launcher.json').read_text());pid=record['pid']
proc=Path('/proc')/str(pid)
if proc.exists():
    args=proc.joinpath('cmdline').read_bytes().split(b'\0')
    assert b'train.py' in args and str(run).encode() in args
    assert proc.joinpath('cwd').resolve()==root/'src'
    commit=json.loads((run/'latest.COMMITTED.json').read_text())
    checkpoint=run/commit['path']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']
    assert (run/'best_development.pt').exists()
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists():break
        # A reaped/zombie process no longer runs model computation.
        if proc.joinpath('stat').read_text().split(') ')[1].startswith('Z'):break
        time.sleep(.1)
    live=proc.exists() and not proc.joinpath('stat').read_text().split(') ')[1].startswith('Z')
    if live:raise RuntimeError('Trainer has not exited; do not escalate or affect other processes')
    result=dict(state='retired_superseded',pid=pid,checkpoint=commit,time=time.time(),
        reason='User prohibited manually named semantic prediction tables; replaced by anonymous_v5',
        other_processes_affected=False)
    (run/'retired.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
else:print(json.dumps(dict(state='already_exited',pid=pid)))
