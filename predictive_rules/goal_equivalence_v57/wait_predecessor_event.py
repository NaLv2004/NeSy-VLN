"""One bounded event subscription to a previously verified supervisor PID."""
import json,os,select,time
from pathlib import Path
pid=2161075
path=Path(f'/proc/{pid}/cmdline');began=time.time()
if path.exists() and path.read_bytes():
    assert b'src_definition_rules_v56/supervisor.py' in path.read_bytes()
    try:handle=os.pidfd_open(pid)
    except ProcessLookupError:handle=None
    if handle is not None:
        try:events=select.select([handle],[],[],55)[0]
        finally:os.close(handle)
        state='predecessor_exited' if events else 'same_supervisor_still_live'
    else:state='predecessor_exited'
else:state='predecessor_exited'
print(json.dumps(dict(pid=pid,state=state,time=time.time(),wait_seconds=time.time()-began,
    event_wait=True,no_gpu_or_training_file_poll=True)))
