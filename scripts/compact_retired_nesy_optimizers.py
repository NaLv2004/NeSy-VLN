"""Retain every model and best full resume state; strip only cold optimizer slots."""
import gc,json,math,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
base=Path('/home/user/xjy/nesy_vla_20261010').resolve();reference=Path('/home/user/xjy/strips_wm_navigation_20261009')
sys.path.insert(0,str(reference))
from stripsnav.io import read_json,load_committed,committed_torch_save,atomic_json,sha256
manifest=base/'cold_nesy_optimizer_compaction.json';records=[];skipped=[]
if manifest.exists():records=read_json(manifest).get('records',[])
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
            if str(run).encode() in p.read_bytes():return True
        except OSError:pass
    return False
for marker in sorted(base.rglob('latest/COMMITTED')):
    folder=marker.parent;run=folder.parent
    # Active and prototype integration checkpoints are kept entirely unchanged.
    if live(run) or run.name=='integration' or 'symbolic_task_v15' in run.parts:continue
    best=run/'best_development'
    if not (best/'COMMITTED').exists():continue
    old=read_json(marker);path=folder/old['checkpoint_file']
    if not path.exists() or path.is_symlink() or path.stat().st_nlink>1 or path.stat().st_size<100*2**20:continue
    best_commit=read_json(best/'COMMITTED');best_file=best/best_commit['checkpoint_file']
    try:
        best_payload=load_committed(best)
        if not isinstance(best_payload.get('optimizer'),dict):continue
        del best_payload
        payload=load_committed(folder)
    except Exception as e:skipped.append({'folder':str(folder),'error':str(e)});continue
    if not isinstance(payload.get('optimizer'),dict):continue
    size=path.stat().st_size;payload['optimizer']=None
    metadata={k:v for k,v in old.items() if k not in ['checkpoint_file','checkpoint_sha256']}
    metadata.update(optimizer_included=False,optimizer_pruned=True,full_optimizer_available_at=str(best),
                    original_full_checkpoint_sha256=old['checkpoint_sha256'])
    assert not live(run)
    committed_torch_save(folder,payload,metadata);actual=load_committed(folder)
    assert same(payload,actual) and read_json(best/'COMMITTED')==best_commit
    assert sha256(best_file)==best_commit['checkpoint_sha256']
    current=read_json(marker);new=folder/current['checkpoint_file']
    assert path.resolve().is_relative_to(base) and path!=new
    row={'folder':str(folder),'old_commit':old,'new_commit':current,'time':time.time(),
        'all_non_optimizer_fields_bit_exact':True,'best_full_resume_preserved':str(best),'released_bytes':size-new.stat().st_size}
    records.append(row);atomic_json(manifest,{'state':'running','pid':os.getpid(),'records':records})
    path.unlink();del actual,payload;gc.collect()
    print(json.dumps({'folder':str(folder),'released_gib':row['released_bytes']/2**30,'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
atomic_json(manifest,{'state':'complete','records':records,'skipped':skipped,'time':time.time(),'free_gib':shutil.disk_usage(base).free/2**30})
