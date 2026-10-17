"""Read-only inventory of immutable checkpoint candidates, not data deletion."""
import json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
files=[]
for parent in ROOT.iterdir():
    if not parent.is_dir() or not parent.name.startswith(('run_','diagnostic_','onpolicy_')):continue
    for p in parent.glob('*.pt'):
        st=p.stat()
        files.append(dict(path=str(p.relative_to(ROOT)),bytes=st.st_size,inode=st.st_ino,links=st.st_nlink,
            commit=json.loads(p.with_suffix('.COMMITTED.json').read_text()) if p.with_suffix('.COMMITTED.json').exists() else None))
record=dict(time=time.time(),free=shutil.disk_usage(ROOT).free,files=files,
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines())
out=ROOT/'diagnostic_onpolicy_refresh_v60';out.mkdir(exist_ok=True)
(out/'storage_inventory.json').write_text(json.dumps(record,indent=2))
groups={}
for f in files:
    c=f['commit'] or {};digest=c.get('sha256')
    if digest:groups.setdefault(digest,[]).append(f)
duplicates=[v for v in groups.values() if len(set(x['inode'] for x in v))>1]
print(json.dumps(dict(free=record['free'],checkpoint_files=len(files),duplicate_groups=duplicates,
    pt_bytes_unique_inodes=sum({f['inode']:f['bytes'] for f in files}.values()),
    gpus=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))))
