import json,os,select,time
from pathlib import Path
OUT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diagnostic_relational_rules_v62')
pid=json.loads((OUT/'launcher.json').read_text())['pid'];p=Path(f'/proc/{pid}/cmdline')
cmd=p.read_bytes() if p.exists() else b''
if cmd:
    assert b'src_relational_rules_v62/probe.py' in cmd
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:fd=None
    if fd is not None:
        try:ready=bool(select.select([fd],[],[],120)[0])
        finally:os.close(fd)
    else:ready=True
else:ready=True
r=dict(time=time.time(),pid=pid,exit_event=ready,status=json.loads((OUT/'status.json').read_text()))
(OUT/'exit_event.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
