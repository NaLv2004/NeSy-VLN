"""Remove obsolete smoke-only intermediate weights, keeping a committed final per run."""
import hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010')
inventory=json.loads((base/'retired_smoke_cleanup_candidates.json').read_text())
groups={}
for row in inventory['candidates']:
    if row['bytes']>1000000:groups.setdefault(row['run'],[]).append(row)
live=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    if p.parent.name==str(os.getpid()):continue
    try:live.append(p.read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    except OSError:pass
records=[];released=0
for name,rows in groups.items():
    run=Path(name).resolve();assert 'smoke' in run.name or run.name=='integration'
    assert not any(str(run) in cmd for cmd in live),'Smoke run unexpectedly live'
    marker=run/'latest/COMMITTED'
    if not marker.exists():continue
    committed=json.loads(marker.read_text());keep=run/'latest'/committed['checkpoint_file']
    assert keep.exists() and hashlib.sha256(keep.read_bytes()).hexdigest()==committed['checkpoint_sha256']
    for row in rows:
        path=Path(row['path']);assert path.resolve().is_relative_to(run)
        if path==keep or path.is_symlink() or path.stat().st_nlink!=1:continue
        assert path.stat().st_size==row['bytes']
        rec={**row,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'preserved_final':str(keep),'time':time.time()}
        records.append(rec)
    # Logs, traces, reports, source, every full training run and the final smoke model remain.
manifest=base/'retired_smoke_pruning_manifest.json';manifest.write_text(json.dumps({'planned':records,'complete':False},indent=2))
for row in records:
    path=Path(row['path']);path.unlink();released+=row['bytes']
    for alias in path.parent.iterdir():
        if alias.is_symlink() and not alias.exists():alias.unlink()
    if path.parent.name!='latest':
        (path.parent/'PRUNED_SMOKE_WEIGHTS.json').write_text(json.dumps(row,indent=2))
manifest.write_text(json.dumps({'removed':records,'complete':True,'released_bytes':released,'free_bytes':shutil.disk_usage(base).free},indent=2))
print(json.dumps({'removed':len(records),'released_gib':released/2**30,'free_gib':shutil.disk_usage(base).free/2**30}))
