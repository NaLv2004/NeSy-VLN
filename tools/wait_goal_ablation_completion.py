import json,os,select,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/goal_condition_ablation_v13')
processes=json.loads((root/'processes.json').read_text());fds={};deadline=time.time()+600
for record in processes:
    p=Path(f"/proc/{record['pid']}/cmdline");cmd=p.read_bytes() if p.exists() else b''
    if cmd:
        assert b'probe_paired_goal_learning.py' in cmd
        fds[os.pidfd_open(record['pid'])]=record['name']
while fds and time.time()<deadline:
    ready=select.select(list(fds),[],[],max(0,deadline-time.time()))[0]
    for fd in ready:os.close(fd);fds.pop(fd)
pending=list(fds.values())
for fd in fds:os.close(fd)
def compact(value):
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items() if k not in ['labels','predictions']}
    return value
results={}
for p in processes:
    out=root/p['name'];result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
    results[p['name']]={'terminal':p['name'] not in pending,'state':result.get('state'),
        'initial':compact(result.get('initial')),'final':compact((result.get('learning') or [None])[-1]),
        'tail':(out/'probe.log').read_text(errors='replace')[-700:]}
summary={'time':time.time(),'results':results};(root/'completion_inspection.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
