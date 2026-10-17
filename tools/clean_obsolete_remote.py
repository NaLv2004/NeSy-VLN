"""Delete superseded latest generations only; preserve all best/snapshot folders."""
import hashlib,json,os,time
from pathlib import Path
root=Path('/home/user/xjy/strips_wm_navigation_20261009').resolve()
items=json.loads((root/'research/obsolete_generations_20261010.json').read_text())
refs=[]
for base in [root/'research',root/'output',root/'configs']:
    for p in base.rglob('*'):
        if p.suffix in {'.json','.md','.py','.html'} and p.stat().st_size<5_000_000 and 'obsolete' not in p.name:
            try: refs.append(p.read_text(errors='replace'))
            except OSError: pass
reference='\n'.join(refs)
deleted=[]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
for row in items:
    p=Path(row['path']); current=Path(row['current'])
    if p.parent.name!='latest' or str(p) in reference or p.name in reference:continue
    assert p.resolve().is_relative_to(root/'runs') and not p.is_symlink()
    marker=json.loads((p.parent/'COMMITTED').read_text())
    assert (p.parent/marker['checkpoint_file']).resolve()==current and p.resolve()!=current
    assert sha(current)==marker['checkpoint_sha256']
    if sha(p) in reference:continue
    p.unlink(); deleted.append(row)
    print(json.dumps({'deleted':str(p),'freed_bytes':sum(x['bytes'] for x in deleted)}),flush=True)
(root/'research/cleanup_latest_generations_20261010.json').write_text(json.dumps({'time':time.time(),'deleted':deleted,'bytes':sum(x['bytes'] for x in deleted)},indent=2))
print('FINISHED',sum(x['bytes'] for x in deleted),flush=True)
