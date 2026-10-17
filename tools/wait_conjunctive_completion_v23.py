import json,os,select,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_conjunctive_v23')
startup=json.loads((root/'startup_inspection.json').read_text());handles={}
for name,job in startup['jobs'].items():
    pid=job['supervisor']['pid'];path=Path(f'/proc/{pid}/cmdline')
    if path.exists() and str(root/'runs'/name).encode() in path.read_bytes():
        try:handles[os.pidfd_open(pid)]=name
        except ProcessLookupError:pass
deadline=time.time()+240
try:
    while handles and time.time()<deadline:
        ready=select.select(list(handles),[],[],max(0,deadline-time.time()))[0]
        for fd in ready:os.close(fd);del handles[fd]
    pending=list(handles.values())
finally:
    for fd in handles:os.close(fd)
read=lambda p:json.loads(p.read_text()) if p.exists() else None
result={'time':time.time(),'pending':pending,'jobs':{}}
for name in startup['jobs']:
    out=root/'runs'/name;pid=startup['jobs'][name]['supervisor']['pid'];path=Path(f'/proc/{pid}/cmdline')
    result['jobs'][name]={'status':read(out/'status.json'),'supervisor':read(out/'supervisor.json'),
        'live':path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else None,
        'commit':read(out/'latest/COMMITTED'),'best_commit':read(out/'best_development/COMMITTED'),
        'best_score':read(out/'best_score.json'),
        'storage_waits':{str(p.relative_to(out)):read(p) for p in out.rglob('WAITING_FOR_STORAGE.json')},
        'evaluations':{step.name:{suite:read(step/suite/'summary.json') for suite in ['development','fixed80']}
                       for step in sorted((out/'evaluations').glob('step_*'))}}
(root/'completion_inspection.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
