"""Wait for a checkpoint event, not repeated training polling."""
import ctypes,hashlib,json,os,select,shutil,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_branch_v17');out=root/'runs/branch_current'
record=json.loads((root/'process.json').read_text());pid=record['supervisor_pid']
p=Path(f'/proc/{pid}/cmdline');assert p.exists() and b'supervise.py' in p.read_bytes()
libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
assert fd>=0;assert libc.inotify_add_watch(fd,os.fsencode(out),0x100|0x80|0x8)>=0
deadline=time.time()+180
def read(p):return json.loads(p.read_text()) if p.exists() else None
try:
    while time.time()<deadline and not (out/'latest/COMMITTED').exists():
        state=read(out/'supervisor.json')
        if state and state.get('state') in ['retry_wait','needs_attention','stopped','complete']:break
        ready=select.select([fd],[],[],max(0,deadline-time.time()))[0]
        if ready:os.read(fd,65536)
finally:os.close(fd)
supervisor=read(out/'supervisor.json');child=(supervisor or {}).get('pid');proc=Path(f'/proc/{child}/cmdline')
commit=read(out/'latest/COMMITTED');verified=None
if commit:
    file=out/'latest'/commit['checkpoint_file'];verified=hashlib.sha256(file.read_bytes()).hexdigest()==commit['checkpoint_sha256']
result={'time':time.time(),'supervisor':supervisor,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
        'status':read(out/'status.json'),'commit':commit,'commit_sha256_verified':verified,
        'initialization':read(out/'initialization.json'),'free_gib':shutil.disk_usage(root).free/2**30,
        'tail':{p.name:p.read_text(errors='replace')[-2600:] for p in sorted(out.glob('launch_*.log'))[-2:]}}
lines=(out/'rank0.jsonl').read_text().splitlines() if (out/'rank0.jsonl').exists() else []
result['last_update']=json.loads(lines[-1]) if lines else None
(root/'startup_inspection.json').write_text(json.dumps(result,indent=2))
result.pop('last_update');print(json.dumps(result,indent=2))
