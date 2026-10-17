"""Find exact byte-identical checkpoint files without altering anything."""
import collections,hashlib,json,time
from pathlib import Path
root=Path('/home/user/xjy/strips_wm_navigation_20261009/runs').resolve()
by_size=collections.defaultdict(list)
for p in root.rglob('*.pt'):
    if p.is_symlink() or not p.is_file():continue
    st=p.stat()
    if st.st_size>10_000_000:by_size[st.st_size].append((p,st.st_ino,st.st_nlink))
groups=[];hashed_bytes=0
for size,rows in by_size.items():
    if len({inode for _,inode,_ in rows})<2:continue
    by_digest=collections.defaultdict(list)
    for p,inode,links in rows:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        hashed_bytes+=size;by_digest[h.hexdigest()].append({'path':str(p),'inode':inode,'links':links})
    for digest,items in by_digest.items():
        if len({x['inode'] for x in items})>1:groups.append({'sha256':digest,'size':size,'files':items})
result={'time':time.time(),'root':str(root),'exact_duplicate_groups':groups,'hashed_bytes':hashed_bytes,
    'potential_saved_bytes':sum((len({x['inode'] for x in g['files']})-1)*g['size'] for g in groups)}
out=Path('/home/user/xjy/nesy_vla_20261010/evidence_archives/prior_exact_duplicates.json')
out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='exact_duplicate_groups'}))
