"""Event wait on the already verified coordinator; no repeated job polling."""
import json,os,select,time
from pathlib import Path
OUT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diagnostic_ensemble_rules_v61')
pid=json.loads((OUT/'evaluation.launch.json').read_text())['pid']
p=Path(f'/proc/{pid}/cmdline')
command=p.read_bytes() if p.exists() else b''
if command:
    assert b'src_ensemble_rules_v61/coordinate_evaluation.py' in command
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:fd=None
    if fd is not None:
        try:ready=bool(select.select([fd],[],[],120)[0])
        finally:os.close(fd)
    else:ready=True
else:ready=True
result=dict(time=time.time(),pid=pid,exit_event=ready,
    status=json.loads((OUT/'evaluation.status.json').read_text()))
(OUT/'exit_event.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
