"""Keep every old final model; remove unused optimizer slots when best is resumable."""
import hashlib,json,os,sys,time,math,gc,shutil
from pathlib import Path
import torch,numpy as np
root=Path('/home/user/xjy/strips_wm_navigation_20261009');sys.path.insert(0,str(root))
from stripsnav.io import read_json,load_committed,committed_torch_save,atomic_json,sha256
manifest=root/'retired_optimizer_compaction.json';records=read_json(manifest).get('records',[]) if manifest.exists() else [];skipped=[]
def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and math.isnan(a):return isinstance(b,float) and math.isnan(b)
    return a==b
def live(run):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:
            if str(run) in p.read_bytes().replace(b'\0',b' ').decode(errors='replace'):return True
        except OSError:pass
    return False
for sidecar in sorted((root/'runs').glob('*/latest/ARCHIVED_CHECKPOINT.json')):
    folder=sidecar.parent;run=folder.parent
    if live(run):continue
    archive=read_json(sidecar);old=read_json(folder/'COMMITTED');best_marker=read_json(run/'best/COMMITTED')
    if 'checkpoint_file' not in old:
        skipped.append({'folder':str(folder),'reason':'Legacy commit retained as a complete lossless archive; no optimizer pruning.'});continue
    best_file=run/'best'/best_marker.get('checkpoint_file','checkpoint.pt')
    if not best_file.exists():continue
    best=load_committed(run/'best')
    if not isinstance(best.get('optimizer'),dict):continue
    del best
    payload=load_committed(folder)
    if not isinstance(payload.get('optimizer'),dict):continue
    optimizer_bytes=sum(t.numel()*t.element_size() for values in payload['optimizer'].get('state',{}).values() for t in values.values() if torch.is_tensor(t))
    payload['optimizer']=None
    assert not live(run)
    metadata={k:v for k,v in old.items() if k not in ['checkpoint_file','checkpoint_sha256']}
    metadata.update(optimizer_included=False,optimizer_pruned=True,original_full_checkpoint_sha256=old['checkpoint_sha256'],
        full_optimizer_available_at=str(run/'best'))
    committed_torch_save(folder,payload,metadata)
    actual=load_committed(folder)
    assert same(payload,actual),'Non-optimizer checkpoint content changed'
    assert read_json(run/'best/COMMITTED')==best_marker and sha256(best_file)==best_marker['checkpoint_sha256']
    current=read_json(folder/'COMMITTED');new_file=folder/current['checkpoint_file'];packed=Path(archive['archive'])
    assert packed.resolve().is_relative_to((root/'runs').resolve()) and packed.is_file() and not packed.is_symlink()
    row={'time':time.time(),'folder':str(folder),'old_commit':old,'new_commit':current,'best_untouched':str(best_file),
        'all_non_optimizer_fields_bit_exact':True,'unused_optimizer_bytes':optimizer_bytes,
        'released_bytes':packed.stat().st_size-new_file.stat().st_size}
    records.append(row);atomic_json(manifest,{'complete':False,'records':records})
    packed.unlink();sidecar.rename(folder/'ARCHIVED_OPTIMIZER_PRUNED.json')
    print(json.dumps({'folder':str(folder),'released_gib':row['released_bytes']/2**30,'free_gib':shutil.disk_usage(root).free/2**30}),flush=True)
    del payload,actual;gc.collect()
atomic_json(manifest,{'complete':True,'records':records,'skipped':skipped,'released_bytes':sum(r['released_bytes'] for r in records)})
print(json.dumps({'files':len(records),'released_gib':sum(r['released_bytes'] for r in records)/2**30,'free_gib':shutil.disk_usage(root).free/2**30}),flush=True)
