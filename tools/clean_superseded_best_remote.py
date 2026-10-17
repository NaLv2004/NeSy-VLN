"""Preserve each committed best checkpoint and all snapshots; remove only its superseded generations."""
import hashlib,json,time
from pathlib import Path

root=Path('/home/user/xjy/strips_wm_navigation_20261009').resolve()
references=[]
for base in [root/'research',root/'output',root/'configs']:
    for p in base.rglob('*'):
        if p.suffix in {'.json','.md','.py','.html'} and p.stat().st_size<5_000_000 and not any(x in p.name for x in ['obsolete','cleanup']):
            try:references.append(p.read_text(errors='replace'))
            except OSError:pass
reference='\n'.join(references)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

deleted=[];preserved=[]
for marker in root.joinpath('runs').rglob('COMMITTED'):
    folder=marker.parent
    if not folder.name.lower().startswith('best'):continue
    data=json.loads(marker.read_text())
    if 'checkpoint_file' not in data:continue
    current=(folder/data['checkpoint_file']).resolve()
    assert current.is_relative_to(root/'runs') and current.is_file()
    assert sha(current)==data['checkpoint_sha256']
    preserved.append(str(current))
    for p in folder.glob('checkpoint.*.pt'):
        if p.is_symlink() or p.resolve()==current:continue
        assert p.resolve().is_relative_to(root/'runs')
        if str(p) in reference or p.name in reference or sha(p) in reference:continue
        assert json.loads(marker.read_text())==data,'Checkpoint changed during cleanup'
        row={'path':str(p),'bytes':p.stat().st_size,'retained_committed':str(current)}
        p.unlink();deleted.append(row)
audit={'time':time.time(),'deleted':deleted,'preserved':preserved,'bytes':sum(x['bytes'] for x in deleted)}
(root/'research/cleanup_superseded_best_20261010.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({'deleted_generations':len(deleted),'freed_gib':audit['bytes']/2**30,'retained_best_checkpoints':len(preserved)}))
