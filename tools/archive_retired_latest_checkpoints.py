"""Losslessly compress cold historical latest weights; retain best weights online.

Every original checkpoint byte is retained and verified by SHA256. A restoration
manifest accompanies each marker; active NeSy checkpoints are never considered.
"""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/strips_wm_navigation_20261009/runs').resolve()
manifest=base.parent/'retired_latest_compression.json'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def commands():
    rows=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:rows.append(p.read_bytes().replace(b'\0',b' ').decode(errors='replace'))
        except OSError:pass
    return rows
records=[];active=commands();released=0
for latest in sorted(base.glob('*/latest')):
    run=latest.parent;best=run/'best/COMMITTED';marker=latest/'COMMITTED'
    if not best.exists() or not marker.exists():continue
    if any(str(run) in cmd for cmd in active):continue
    record=json.loads(marker.read_text());best_record=json.loads(best.read_text())
    source=latest/record.get('checkpoint_file','checkpoint.pt')
    best_file=best.parent/best_record.get('checkpoint_file','checkpoint.pt')
    if not source.exists() or source.is_symlink() or source.stat().st_nlink!=1 or not best_file.exists():continue
    assert source.resolve().is_relative_to(base) and source.suffix=='.pt'
    assert digest(source)==record['checkpoint_sha256'] and digest(best_file)==best_record['checkpoint_sha256']
    dest=source.with_suffix('.pt.zst');temporary=dest.with_suffix('.partial')
    assert not dest.exists() and not temporary.exists()
    subprocess.run(['zstd','-q','-3','-T2',str(source),'-o',str(temporary)],check=True)
    with subprocess.Popen(['zstd','-q','-d','-c',str(temporary)],stdout=subprocess.PIPE) as process:
        h=hashlib.sha256();size=0
        for block in iter(lambda:process.stdout.read(8*1024*1024),b''):h.update(block);size+=len(block)
        assert process.wait()==0
    assert size==source.stat().st_size and h.hexdigest()==record['checkpoint_sha256']
    assert json.loads(marker.read_text())==record and not any(str(run) in c for c in commands())
    temporary.replace(dest)
    item={'time':time.time(),'source':str(source),'archive':str(dest),'original_sha256':h.hexdigest(),
        'original_bytes':size,'compressed_bytes':dest.stat().st_size,'best_preserved':str(best_file),
        'restore_argv':['zstd','-d','-k',str(dest),'-o',str(source)]}
    (latest/'ARCHIVED_CHECKPOINT.json').write_text(json.dumps(item,indent=2))
    records.append(item);manifest.write_text(json.dumps({'complete':False,'records':records},indent=2))
    source.unlink();released+=size-dest.stat().st_size
    print(json.dumps({'run':run.name,'released_gib':released/2**30,'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
manifest.write_text(json.dumps({'complete':True,'records':records,'released_bytes':released},indent=2))
print(json.dumps({'files':len(records),'released_gib':released/2**30,'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
