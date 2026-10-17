import json,os,select,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/counterfactual_branch_v20');processes=json.loads((root/'processes.json').read_text())
fds={};deadline=time.time()+600
for record in processes:
    p=Path(f"/proc/{record['pid']}/cmdline")
    if p.exists() and p.read_bytes():
        assert b'probe_paired_goal_learning.py' in p.read_bytes()
        fds[os.pidfd_open(record['pid'])]=record['name']
while fds and time.time()<deadline:
    ready=select.select(list(fds),[],[],max(0,deadline-time.time()))[0]
    for fd in ready:os.close(fd);fds.pop(fd)
pending=list(fds.values())
for fd in fds:os.close(fd)
def strip(x):return {k:strip(v) for k,v in x.items() if k not in ['predictions','labels']} if isinstance(x,dict) else x
results={}
for record in processes:
    out=root/record['name'];value=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
    results[record['name']]={'terminal':record['name'] not in pending,'state':value.get('state'),
        'initial':strip(value.get('initial')),'last':strip((value.get('learning') or [None])[-1]),
        'tail':(out/'probe.log').read_text(errors='replace')[-700:]}
result={'time':time.time(),'results':results};(root/'completion_inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

