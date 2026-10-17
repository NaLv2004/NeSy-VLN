"""Preserve every immutable archive path and byte while sharing identical files."""
import hashlib,json,os,shutil,time
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
groups=defaultdict(list)
for folder in ROOT.iterdir():
    if not folder.is_dir() or folder.name=='run_goal_aware_symbols_v52':continue
    if folder.name.startswith('run_'):
        candidates=list(folder.glob('checkpoint_*.pt'))+list(folder.glob('dev_*/program_model_rank*.pt'))
    elif folder.name.startswith('diagnostic_'):
        candidates=list(folder.glob('source.pt'))+list(folder.glob('*/program_model_rank*.pt'))
    else:continue
    for path in candidates:
        if path.is_symlink() or not path.is_file():continue
        assert path.resolve().is_relative_to(ROOT)
        groups[path.stat().st_size].append(path)
before=shutil.disk_usage(ROOT).free;changes=[];known_inodes={}
for size,paths in groups.items():
    if len(paths)<2:continue
    by_sha={}
    for path in paths:
        st=path.stat();inode=(st.st_dev,st.st_ino)
        digest=known_inodes.get(inode)
        if digest is None:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
            digest=h.hexdigest();known_inodes[inode]=digest
        if digest not in by_sha:by_sha[digest]=path;continue
        source=by_sha[digest];ss=source.stat()
        if (ss.st_dev,ss.st_ino)==inode:continue
        assert ss.st_size==size and ss.st_dev==st.st_dev
        # Atomically replace an immutable copy. Content and existing recipe
        # digests remain unchanged; neither model weights nor history is lost.
        temp=path.with_name(path.name+'.dedup_link')
        assert not temp.exists()
        os.link(source,temp);os.replace(temp,path)
        assert os.path.samefile(source,path)
        changes.append(dict(path=str(path),source=str(source),sha256=digest,bytes=size,
                            freed_inode=st.st_nlink==1))
record=dict(time=time.time(),free_before=before,free_after=shutil.disk_usage(ROOT).free,
            changes=changes,all_paths_and_bytes_preserved=True,active_run_untouched=True)
(ROOT/'archive_deduplication_v53.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
