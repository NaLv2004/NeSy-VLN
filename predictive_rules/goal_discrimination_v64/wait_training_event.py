"""One OS process-exit wait, not repeated training polling."""
import os,shlex,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from predictive_remote import connect

inspection=Path(__file__).with_name('inspect_training_status.py').read_text()
remote='''import os,select,json,time
from pathlib import Path
pid=2233147
p=Path(f'/proc/{pid}/cmdline')
if p.exists():
    command=p.read_bytes()
    assert b'src_goal_discrimination_v64/supervise_training.py' in command,command
    fd=os.pidfd_open(pid)
    print(json.dumps(dict(event='waiting_on_verified_process',pid=pid,time=time.time())),flush=True)
    try:ready=select.select([fd],[],[],600)[0]
    finally:os.close(fd)
    if not ready:
        print(json.dumps(dict(event='wait_timeout_process_still_live',pid=pid,time=time.time())),flush=True)
        raise SystemExit(0)
'''+inspection
client=connect()
try:
    _,out,err=client.exec_command('python3 -c '+shlex.quote(remote),timeout=660)
    for line in iter(out.readline,''):print(line,end='',flush=True)
    error=err.read().decode()
    if error:print(error,flush=True)
    raise SystemExit(out.channel.recv_exit_status())
finally:client.close()
