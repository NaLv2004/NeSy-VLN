import json,os,select,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_task_v15');run=root/'runs/paired_tasks'
record=json.loads((root/'process.json').read_text());pid=record['supervisor_pid'];proc=Path(f'/proc/{pid}/cmdline')
pending=False
if proc.exists() and proc.read_bytes():
    assert b'supervise.py' in proc.read_bytes() and str(root).encode() in proc.read_bytes()
    fd=os.pidfd_open(pid)
    try:pending=not bool(select.select([fd],[],[],600)[0])
    finally:os.close(fd)
def read(p):return json.loads(p.read_text()) if p.exists() else None
result={'time':time.time(),'pending':pending,'pid':pid,
    'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
    'supervisor':read(run/'supervisor.json'),'status':read(run/'status.json'),'commit':read(run/'latest/COMMITTED'),
    'evaluations':{step.name:{suite:read(step/suite/'summary.json') for suite in ['development','fixed80']}
        for step in sorted((run/'evaluations').glob('step_*'))}}
(root/'completion_inspection.json').write_text(json.dumps(result,indent=2))
result['evaluations']={step:{suite:summary.get('successes') if summary else None for suite,summary in entries.items()} for step,entries in result['evaluations'].items()}
print(json.dumps(result,indent=2))
