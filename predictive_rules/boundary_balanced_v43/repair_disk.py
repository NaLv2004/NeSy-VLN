"""Remove only obsolete generated dev4000 tables after terminal-job verification."""
import hashlib,json,shutil,time
from pathlib import Path

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015').resolve()
run=ROOT/'run_boundary_balanced_v43'
for pid in (2050991,2050992,2050997,2050998):
    proc=Path(f'/proc/{pid}/cmdline')
    assert not proc.exists() or not proc.read_bytes(),f'Owned process still live: {pid}'
commit=json.loads((run/'latest.COMMITTED.json').read_text())
assert commit['update']==4500
assert hashlib.sha256((run/commit['filename']).read_bytes()).hexdigest()==commit['sha256']
best=json.loads((run/'best.COMMITTED.json').read_text())
folder=(run/'dev_004000').resolve()
assert folder.parent==run and best['update']!=4000 and (folder/'summary.json').is_file()
files=sorted(folder.glob('*/program_*.npz'))
assert files,'No obsolete tables remain'
for file in files:
    assert file.resolve().is_relative_to(folder) and not file.is_symlink()
    assert (file.parent/'trace.json').is_file()
record=dict(time=time.time(),reason='Disk-full recovery. Remove only obsolete generated dev4000 tables; preserve all checkpoints, best full tables, traces, images, and incomplete dev4500.',
    checkpoint_verified=commit,files=[dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size) for p in files],
    before=shutil.disk_usage(ROOT)._asdict())
print(json.dumps(dict(phase='validated_before_removal',**record)),flush=True)
# The filesystem has zero free bytes: release one already validated obsolete
# generated table before persisting the manifest. The full manifest also goes
# to the caller before any deletion, so it survives a second disk failure.
for file in files:file.unlink()
manifest=folder/'program_table_retention_disk_repair.json'
record['after']=shutil.disk_usage(ROOT)._asdict()
record['bytes_removed']=sum(p['bytes'] for p in record['files'])
manifest.write_text(json.dumps(record,indent=2))
print(json.dumps(dict(phase='completed',removed=record['bytes_removed'],free=record['after']['free'],checkpoint=commit)),flush=True)
