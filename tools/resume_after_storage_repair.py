import hashlib,json,os,signal,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');path=base/'storage_hold_20261011.json';record=json.loads(path.read_text())
assert record['state']=='holding' and record['extra_workers_held']
assert shutil.disk_usage(base).free>8*2**30
roots=['symbolic_robust_v7/runs/robust_multieffect','symbolic_joint_v9/runs/joint',
       'symbolic_vision_v8/runs/frozen_convnext','symbolic_vision_v8/runs/frozen_qwen']
commits={}
for relative in roots:
    folder=base/relative/'latest';marker=json.loads((folder/'COMMITTED').read_text())
    assert hashlib.sha256((folder/marker['checkpoint_file']).read_bytes()).hexdigest()==marker['checkpoint_sha256']
    commits[relative]=marker
for row in sorted(record['processes'],key=lambda x:x['supervisor']):
    cmd=Path(f"/proc/{row['pid']}/cmdline")
    if not cmd.exists():continue
    assert cmd.read_bytes().replace(b'\0',b' ').decode()==row['cmd']
    if row['supervisor']:os.kill(row['pid'],signal.SIGCONT)
    else:os.killpg(row['pgid'],signal.SIGCONT)
record.update(state='resume_requested',resume_time=time.time(),verified_commits=commits,
    free_gib=shutil.disk_usage(base).free/2**30,checkpoint_space_guard='Wait before writing below3GiB; installed and tested')
path.write_text(json.dumps(record,indent=2));print(json.dumps({'state':record['state'],'free_gib':record['free_gib'],
    'verified_updates':{k:v['epoch'] for k,v in commits.items()}}))
