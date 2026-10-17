"""Retain every committed/named model; prune unreferenced rollback leftovers of stopped runs."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
protected=set()
for marker in base.rglob('COMMITTED'):
    try:
        name=json.loads(marker.read_text()).get('checkpoint_file')
        if name:protected.add((marker.parent/name).resolve())
    except (ValueError,OSError):pass
live=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    if p.parent.name==str(os.getpid()):continue
    try:live.append(p.read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    except OSError:pass
records=[]
for marker in base.rglob('COMMITTED'):
    folder=marker.parent
    if folder.name not in ['latest','best_development']:continue
    run=folder.parent
    if any(str(run) in cmd and ('scripts/train.py' in cmd or 'scripts/supervise.py' in cmd) for cmd in live):continue
    current=json.loads(marker.read_text());keep=folder/current['checkpoint_file']
    if not keep.exists():continue
    for p in folder.glob('checkpoint.e*.pt'):
        if p.resolve() in protected or p.is_symlink() or p.stat().st_nlink>1:continue
        assert p.resolve().is_relative_to(base) and p!=keep
        records.append({'path':str(p),'bytes':p.stat().st_size,'preserved_committed':str(keep)})
manifest=base/'retired_rollback_pruning.json'
if '--apply' not in sys.argv:
    manifest.write_text(json.dumps({'candidates':records,'time':time.time()},indent=2));print(json.dumps({'candidates':records,'gib':sum(r['bytes'] for r in records)/2**30},indent=2))
else:
    reviewed=json.loads(manifest.read_text())['candidates'];assert reviewed==records,'Plan changed; inspect again'
    for row in records:
        p=Path(row['path']);row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'removed':records,'complete':False,'time':time.time()},indent=2))
    for row in records:Path(row['path']).unlink()
    manifest.write_text(json.dumps({'removed':records,'complete':True,'time':time.time()},indent=2))
    print(json.dumps({'files':len(records),'released_gib':sum(r['bytes'] for r in records)/2**30,'free_gib':shutil.disk_usage(base).free/2**30}))
