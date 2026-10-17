"""Discard unreferenced atomic-save rollback copies, preserving all committed models."""
import hashlib,json,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
protected=set()
for m in base.rglob('COMMITTED'):
    r=json.loads(m.read_text());protected.add((m.parent/r.get('checkpoint_file','checkpoint.pt')).resolve())
records=[]
for row in json.loads((base/'degraded_runs_retirement.json').read_text()):
    run=Path(row['run']).resolve();assert run.is_relative_to(base) and (run/'STOP').exists()
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=proc.read_bytes()
        except OSError:continue
        assert not (str(run).encode() in cmd and (b'scripts/train.py' in cmd or b'scripts/supervise.py' in cmd))
    for folder in [run/'latest',run/'best_development']:
        marker=folder/'COMMITTED'
        if not marker.exists():continue
        commit=json.loads(marker.read_text());keep=folder/commit['checkpoint_file']
        assert hashlib.sha256(keep.read_bytes()).hexdigest()==commit['checkpoint_sha256']
        for p in folder.glob('checkpoint.e*.pt'):
            if p.resolve() in protected or p.is_symlink() or p.stat().st_nlink!=1:continue
            assert p.resolve().is_relative_to(folder.resolve())
            records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'preserved':str(keep)})
manifest=base/'retired_uncommitted_pruning_20261011.json'
manifest.write_text(json.dumps({'state':'verified_plan','records':records,'time':time.time()},indent=2))
for r in records:Path(r['path']).unlink()
result={'state':'complete','records':records,'time':time.time(),'free_gib':shutil.disk_usage(base).free/2**30}
manifest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
