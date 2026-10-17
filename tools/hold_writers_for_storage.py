import json,os,signal,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'storage_hold_20261011.json'
assert not out.exists(),'Inspect an existing hold before repeating'
roots=[base/'symbolic_robust_v7/runs/robust_multieffect',base/'symbolic_joint_v9/runs/joint',
       base/'symbolic_vision_v8/runs/frozen_convnext',base/'symbolic_vision_v8/runs/frozen_qwen']
rows=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:cmd=path.read_bytes().replace(b'\0',b' ').decode();pid=int(path.parent.name)
    except (OSError,ValueError):continue
    if pid==os.getpid() or not any(str(root) in cmd for root in roots):continue
    if 'scripts/supervise.py' in cmd or 'torch.distributed.run' in cmd:
        rows.append({'pid':pid,'pgid':os.getpgid(pid),'cmd':cmd,'supervisor':'scripts/supervise.py' in cmd})
record={'state':'holding','time':time.time(),'reason':'Disk exhausted during checkpoint save; preserve committed state while reclaiming cold storage','processes':rows}
out.write_text(json.dumps(record,indent=2))
for row in sorted(rows,key=lambda x:not x['supervisor']):
    if not Path(f"/proc/{row['pid']}/cmdline").exists():continue
    if row['supervisor']:os.kill(row['pid'],signal.SIGSTOP)
    else:
        assert row['pgid']==row['pid'] and row['pgid']!=os.getpgrp()
        os.killpg(row['pgid'],signal.SIGSTOP)
print(json.dumps(record,indent=2))
