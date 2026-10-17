"""Observe a specific collection coordinator exit without status polling."""
import json,os,select,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diverse_history_v70')
pid=json.loads((out/'launch.json').read_text())['pid']
try:fd=os.pidfd_open(pid)
except ProcessLookupError:fd=None
event=True
if fd is not None:
    ready,_,_=select.select([fd],[],[],45);event=bool(ready);os.close(fd)
record=dict(time=time.time(),pid=pid,exit_event=event,
    actual_command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{pid}/cmdline').exists() else '',
    state=json.loads((out/'status.json').read_text()))
print(json.dumps(record))
