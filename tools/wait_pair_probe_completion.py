"""Wait on Linux process termination; no repeated training/status-file polling."""
import json,os,select,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/paired_goal_probe_v12')
record=json.loads((root/'probe_process.json').read_text());pid=record['pid']
p=Path(f'/proc/{pid}/cmdline');cmd=p.read_bytes() if p.exists() else b''
terminal=not cmd
if cmd:
    assert b'probe_paired_goal_learning.py' in cmd
    fd=os.pidfd_open(pid)
    try:terminal=bool(select.select([fd],[],[],600)[0])
    finally:os.close(fd)
result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None
def compact(value):
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items() if k not in ['labels','predictions','dataset_provenance']}
    if isinstance(value,list):return [compact(v) for v in value]
    return value
print(json.dumps({'time':time.time(),'pid':pid,'process_terminal':terminal,
    'state':(result or {}).get('state'),'initial':compact((result or {}).get('initial')),
    'final':compact(((result or {}).get('learning') or [None])[-1]),
    'tail':(root/'probe.log').read_text(errors='replace')[-2000:]},indent=2))
