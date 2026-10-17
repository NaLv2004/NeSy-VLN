import ctypes,hashlib,json,os,select,shutil,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_teacher_v25');records=json.loads((root/'processes.json').read_text())
libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);assert fd>=0
for record in records:
    pid=record['supervisor_pid'];proc=Path(f'/proc/{pid}/cmdline')
    assert proc.exists() and b'supervise.py' in proc.read_bytes()
    assert libc.inotify_add_watch(fd,os.fsencode(root/'runs'/record['name']),0x100|0x80|0x8)>=0
def read(path):return json.loads(path.read_text()) if path.exists() else None
deadline=time.time()+240
try:
    while time.time()<deadline:
        ready_count=0;bad=False
        for record in records:
            out=root/'runs'/record['name'];ready_count+=int((out/'latest/COMMITTED').exists())
            supervisor=read(out/'supervisor.json') or {}
            bad=bad or supervisor.get('state') in ['retry_wait','needs_attention','stopped']
        if ready_count==len(records) or bad:break
        if select.select([fd],[],[],max(0,deadline-time.time()))[0]:os.read(fd,65536)
finally:os.close(fd)
result={'time':time.time(),'free_gib':shutil.disk_usage(root).free/2**30,'jobs':{}}
for record in records:
    out=root/'runs'/record['name'];commit=read(out/'latest/COMMITTED');supervisor=read(out/'supervisor.json')
    pid=(supervisor or {}).get('pid');proc=Path(f'/proc/{pid}/cmdline')
    lines=(out/'rank0.jsonl').read_text().splitlines() if (out/'rank0.jsonl').exists() else []
    result['jobs'][record['name']]={'commit':commit,'commit_verified':bool(commit) and
        hashlib.sha256((out/'latest'/commit['checkpoint_file']).read_bytes()).hexdigest()==commit['checkpoint_sha256'],
        'supervisor':supervisor,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
        'status':read(out/'status.json'),'last_update':json.loads(lines[-1]) if lines else None,
        'tail':{p.name:p.read_text(errors='replace')[-1500:] for p in sorted(out.glob('launch_*.log'))[-1:]}}
(root/'startup_inspection.json').write_text(json.dumps(result,indent=2))
for value in result['jobs'].values():
    row=value['last_update']
    if row:value['last_update']={k:row[k] for k in ['update','seconds','gradient_norm','gpu_peak_gib','losses','symbolic_audit']}
print(json.dumps(result,indent=2))

