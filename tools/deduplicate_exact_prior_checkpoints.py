"""Keep all checkpoint paths and bytes; coalesce only verified identical files."""
import hashlib,json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010/evidence_archives')
inventory=json.loads((base/'prior_exact_duplicates.json').read_text());root=Path(inventory['root']).resolve()
commands=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
records=[]
for group in inventory['exact_duplicate_groups']:
    paths=[Path(row['path']) for row in group['files']]
    assert all(p.resolve().is_relative_to(root) and not p.is_symlink() for p in paths)
    assert not any(str(p.parent) in command and 'train' in command for p in paths for command in commands)
    canonical=paths[0];assert sha(canonical)==group['sha256']
    for path in paths[1:]:
        if path.stat().st_ino==canonical.stat().st_ino:continue
        assert sha(path)==group['sha256']
        before=path.stat();temporary=path.with_name(path.name+'.identical_link')
        assert not temporary.exists();os.link(canonical,temporary)
        assert path.stat().st_ino==before.st_ino and path.stat().st_mtime_ns==before.st_mtime_ns
        os.replace(temporary,path)
        assert path.stat().st_ino==canonical.stat().st_ino
        records.append({'path':str(path),'canonical':str(canonical),'sha256':group['sha256'],'bytes':before.st_size})
result={'time':time.time(),'linked':records,'bytes_coalesced':sum(x['bytes'] for x in records),
    'preserved':'All original file paths and SHA256 contents are identical; no model generation deleted.'}
(base/'prior_exact_deduplication.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
