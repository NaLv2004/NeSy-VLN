"""Share byte-identical cold checkpoints while retaining every original path."""
import hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve();reference=Path('/home/user/xjy/strips_wm_navigation_20261009').resolve()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
groups={};seen=set();records=[]
commands=[]
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    if proc.parent.name==str(os.getpid()):continue
    try:commands.append(proc.read_bytes())
    except OSError:pass
for root in [base,reference/'runs']:
    for marker in root.rglob('COMMITTED'):
        if 'evidence_archives' in marker.parts:continue
        try:record=json.loads(marker.read_text())
        except (ValueError,OSError):continue
        if 'checkpoint_sha256' not in record:continue
        folder=marker.parent;path=folder/record.get('checkpoint_file','checkpoint.pt')
        if not path.is_file() or path.is_symlink() or path in seen:continue
        if any(str(folder.parent).encode() in command for command in commands):continue
        if 'symbolic_paired_v12' in path.parts:continue
        assert path.resolve().is_relative_to(root)
        seen.add(path);groups.setdefault((record['checkpoint_sha256'],path.stat().st_size),[]).append((path,marker,record))
for (digest,size),entries in groups.items():
    if len(entries)<2:continue
    entries.sort(key=lambda e:str(e[0]));source=entries[0][0]
    if sha(source)!=digest:raise ValueError('Source checkpoint hash mismatch')
    for target,marker,record in entries[1:]:
        if target.stat().st_ino==source.stat().st_ino:continue
        assert target.stat().st_dev==source.stat().st_dev and sha(target)==digest
        assert json.loads(marker.read_text())==record
        before=target.stat();temporary=target.with_name(target.name+f'.dedup.{os.getpid()}')
        os.link(source,temporary);temporary.replace(target)
        assert target.stat().st_ino==source.stat().st_ino
        records.append({'source':str(source),'target':str(target),'sha256':digest,
            'bytes':size,'released_bytes':size if before.st_nlink==1 else 0,'all_original_paths_retained':True})
result={'time':time.time(),'state':'complete','records':records,'released_bytes':sum(r['released_bytes'] for r in records),
        'free_gib':shutil.disk_usage(base).free/2**30}
(base/'cold_identical_checkpoint_deduplication.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'files':len(records),'released_gib':result['released_bytes']/2**30,'free_gib':result['free_gib']},indent=2))
