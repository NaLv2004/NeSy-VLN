"""Event-driven wait for the exact collector; no periodic process polling."""
import json
import os
from pathlib import Path
import select
import time
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/onpolicy_successors_v40')
status=json.loads((root/'status.json').read_text())
pid=status['pid'];path=Path(f'/proc/{pid}/cmdline')
if path.exists() and b'collect_onpolicy.py' in path.read_bytes():
    descriptor=os.pidfd_open(pid)
    try:
        ready=bool(select.select([descriptor],[],[],150)[0])
    finally:
        os.close(descriptor)
else:
    ready=True
print(json.dumps(dict(process_exited=ready,time=time.time(),status=json.loads((root/'status.json').read_text()))))
