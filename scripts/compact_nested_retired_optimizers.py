"""Keep all models and each best full resume state; compact retired latest optimizers."""
import gc,json,math,os,signal,sys,time,shutil
from pathlib import Path
import torch,numpy as np
root=Path('/home/user/xjy/strips_wm_navigation_20261009');sys.path.insert(0,str(root))
from stripsnav.io import read_json,load_committed,committed_torch_save,atomic_json,sha256
manifest=root/'nested_optimizer_compaction.json';records=[];skipped=[]
if manifest.exists():records=read_json(manifest).get('records',[])
# Some historical latest/best files are already hard-linked. Reuse the shared
# full checkpoint instead of materializing a second optimizer-free copy.
for record in records:
    if record.get('reverted_to_shared_full_checkpoint'):continue
    folder=Path(record['folder']);best_folder=Path(record['best_full_optimizer_preserved'])
    best_marker=read_json(best_folder/'COMMITTED');old=record['old_commit']
    if old['checkpoint_sha256']!=best_marker['checkpoint_sha256']:continue
    current=read_json(folder/'COMMITTED')
    assert current==record['new_commit']
    source=best_folder/best_marker.get('checkpoint_file','checkpoint.pt');assert sha256(source)==old['checkpoint_sha256']
    restored=folder/old.get('checkpoint_file','checkpoint.pt')
    if restored.is_symlink():restored.unlink()
    if not restored.exists():os.link(source,restored)
    assert sha256(restored)==old['checkpoint_sha256']
    atomic_json(folder/'COMMITTED',old)
    alias=folder/'checkpoint.pt'
    if alias!=restored:
        if alias.is_symlink():alias.unlink()
        alias.symlink_to(restored.name)
    compact=folder/current['checkpoint_file']
    if compact.resolve()!=restored.resolve():compact.unlink()
    record.update(reverted_to_shared_full_checkpoint=True,released_bytes=0,
        note='Already shared with best; retain original full checkpoint without duplicate storage')
atomic_json(manifest,{'state':'running','records':records})
def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and math.isnan(a):return isinstance(b,float) and math.isnan(b)
    return a==b
def commands():
    rows=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:rows.append((int(p.parent.name),p.read_bytes().replace(b'\0',b' ').decode(errors='replace')))
        except OSError:pass
    return rows
for pid,cmd in commands():
    if cmd.startswith('python3 -c ') and 'Remove obsolete atomic-save generations; retain every committed checkpoint.' in cmd:
        os.kill(pid,signal.SIGTERM)
for marker in sorted((root/'runs').rglob('latest/COMMITTED')):
    folder=marker.parent;run=folder.parent;best_folder=run/'best'
    if not (best_folder/'COMMITTED').exists():continue
    if any(str(run) in c for _,c in commands()):continue
    old=read_json(marker);old_file=folder/old.get('checkpoint_file','checkpoint.pt')
    if not old_file.exists() or 'checkpoint_file' not in old or old_file.stat().st_nlink>1:continue
    best_marker=read_json(best_folder/'COMMITTED')
    try:best=load_committed(best_folder)
    except Exception as error:
        skipped.append({'folder':str(folder),'reason':'Existing best could not be verified; all files retained','error':str(error)});continue
    if not isinstance(best.get('optimizer'),dict):continue
    del best
    try:payload=load_committed(folder)
    except Exception as error:
        skipped.append({'folder':str(folder),'reason':'Existing latest could not be verified; all files retained','error':str(error)});continue
    if not isinstance(payload.get('optimizer'),dict):continue
    size=old_file.stat().st_size;payload['optimizer']=None
    metadata={k:v for k,v in old.items() if k not in ['checkpoint_file','checkpoint_sha256']}
    metadata.update(optimizer_included=False,optimizer_pruned=True,original_full_checkpoint_sha256=old['checkpoint_sha256'],
        full_optimizer_available_at=str(best_folder))
    committed_torch_save(folder,payload,metadata);actual=load_committed(folder)
    assert same(payload,actual) and read_json(best_folder/'COMMITTED')==best_marker
    assert sha256(best_folder/best_marker.get('checkpoint_file','checkpoint.pt'))==best_marker['checkpoint_sha256']
    current=read_json(marker);new_file=folder/current['checkpoint_file']
    assert old_file.resolve().is_relative_to(root/'runs') and old_file.resolve()!=new_file.resolve()
    row={'time':time.time(),'folder':str(folder),'old_commit':old,'new_commit':current,
        'best_full_optimizer_preserved':str(best_folder),'all_non_optimizer_fields_bit_exact':True,
        'released_bytes':size-new_file.stat().st_size}
    records.append(row);atomic_json(manifest,{'state':'running','records':records})
    old_file.unlink()
    print(json.dumps({'folder':str(folder),'released_gib':row['released_bytes']/2**30,
        'free_gib':shutil.disk_usage(root).free/2**30}),flush=True)
    del actual,payload;gc.collect()
atomic_json(manifest,{'state':'complete','records':records,'skipped':skipped,'released_bytes':sum(r['released_bytes'] for r in records),
    'free_gib':shutil.disk_usage(root).free/2**30})
print(json.dumps({'files':len(records),'released_gib':sum(r['released_bytes'] for r in records)/2**30,
    'free_gib':shutil.disk_usage(root).free/2**30}),flush=True)
