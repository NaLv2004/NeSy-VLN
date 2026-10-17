"""Remove obsolete atomic-save generations; retain every committed checkpoint.

Only retired STRIPS runs are eligible. Full SHA verification of the currently
committed file precedes deletion of any other generation in that same folder.
"""
import hashlib,json,os,time,shutil
from pathlib import Path
base=Path('/home/user/xjy/strips_wm_navigation_20261009/runs').resolve()
output=base.parent/'unreferenced_generation_cleanup.json';records=[]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def commands():
    result=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:result.append(p.read_bytes().replace(b'\0',b' ').decode(errors='replace'))
        except OSError:pass
    return result
active=commands()
for marker in sorted(base.rglob('COMMITTED')):
    folder=marker.parent;run=base/folder.relative_to(base).parts[0]
    if any(str(run) in c for c in active):continue
    metadata=json.loads(marker.read_text());name=metadata.get('checkpoint_file','checkpoint.pt')
    current=folder/name
    if not current.exists() or current.is_symlink():continue
    if 'checkpoint_sha256' not in metadata:continue
    assert current.resolve().is_relative_to(base)
    assert digest(current)==metadata['checkpoint_sha256']
    keep=current.resolve();alias=folder/'checkpoint.pt'
    if alias.is_symlink() and alias.resolve()!=keep:continue
    candidates=[p for p in folder.glob('checkpoint.*.pt') if p.resolve()!=keep and not p.is_symlink()]
    candidates+=[p for p in folder.glob('checkpoint.*.tmp') if not p.is_symlink()]
    if not candidates:continue
    assert json.loads(marker.read_text())==metadata and not any(str(run) in c for c in commands())
    for path in candidates:
        assert path.resolve().is_relative_to(base) and path.parent==folder and path.resolve()!=keep
        row={'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'committed_file_retained':str(current),
            'committed_sha256':metadata['checkpoint_sha256'],'time':time.time()}
        # Preserve an audit entry before unlinking the obsolete generation.
        records.append(row);output.write_text(json.dumps({'state':'running','records':records},indent=2))
        path.unlink()
    assert digest(current)==metadata['checkpoint_sha256'] and json.loads(marker.read_text())==metadata
    print(json.dumps({'folder':str(folder),'obsolete_files':len(candidates),
        'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
output.write_text(json.dumps({'state':'complete','records':records,'removed_bytes':sum(r['bytes'] for r in records),
    'every_committed_checkpoint_preserved':True,'free_gib':shutil.disk_usage(base).free/2**30},indent=2))
print(json.dumps({'removed_gib':sum(r['bytes'] for r in records)/2**30,'free_gib':shutil.disk_usage(base).free/2**30}))
