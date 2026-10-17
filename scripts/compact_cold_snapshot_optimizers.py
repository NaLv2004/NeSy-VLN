"""Remove optimizer slots only from cold secondary snapshots; retain all weights.

Every run keeps its verified full best checkpoint. Live runs and linked files
are excluded. Replacement is committed only after a complete bit-exact reload.
"""
import gc,json,math,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
reference=Path('/home/user/xjy/strips_wm_navigation_20261009').resolve()
sys.path.insert(0,str(reference))
from stripsnav.io import read_json,load_committed,atomic_json,sha256
manifest=base/'cold_secondary_optimizer_compaction.json';records=[];skipped=[]
if manifest.exists():records=read_json(manifest).get('records',[])
def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and math.isnan(a):return isinstance(b,float) and math.isnan(b)
    return a==b
def live(run):
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name==str(os.getpid()):continue
        try:
            if str(run).encode() in path.read_bytes():return True
        except OSError:pass
    return False
markers=[]
for root in [base,reference/'runs']:
    for marker in root.rglob('COMMITTED'):
        folder=marker.parent
        if folder.name=='latest' or folder.name.startswith('step_'):markers.append((root,marker))
for root,marker in sorted(markers,key=lambda pair:str(pair[1])):
    folder=marker.parent;run=folder.parent
    if live(run) or any(name in run.parts for name in ['symbolic_task_v15','symbolic_branch_v17','symbolic_paired_v12']):continue
    best=next((run/name for name in ['best_development','best'] if (run/name/'COMMITTED').exists()),None)
    if best is None:continue
    old=read_json(marker)
    if 'checkpoint_sha256' not in old:continue
    path=folder/old.get('checkpoint_file','checkpoint.pt')
    if not path.exists() or path.is_symlink() or path.stat().st_nlink>1:continue
    best_commit=read_json(best/'COMMITTED')
    try:
        saved=load_committed(best)
        if not isinstance(saved.get('optimizer'),dict):continue
        del saved;payload=load_committed(folder)
    except Exception as error:skipped.append({'folder':str(folder),'error':str(error)});continue
    if not isinstance(payload.get('optimizer'),dict):del payload;continue
    old_size=path.stat().st_size;payload['optimizer']=None
    if shutil.disk_usage(root).free<600*2**20:raise RuntimeError('Not enough space for verified replacement')
    assert not live(run) and read_json(marker)==old
    replacement=folder/f'checkpoint.optimizer_pruned.{time.time_ns()}.pt';temporary=replacement.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    actual=torch.load(temporary,map_location='cpu',weights_only=False)
    assert same(payload,actual) and read_json(best/'COMMITTED')==best_commit and read_json(marker)==old
    assert not live(run)
    temporary.replace(replacement)
    metadata={k:v for k,v in old.items() if k not in ['checkpoint_file','checkpoint_sha256']}
    metadata.update(optimizer_included=False,optimizer_pruned=True,full_optimizer_available_at=str(best),
                    original_full_checkpoint_sha256=old['checkpoint_sha256'])
    new=dict(metadata,checkpoint_file=replacement.name,checkpoint_sha256=sha256(replacement))
    atomic_json(marker,new);atomic_json(folder/'metadata.json',metadata)
    alias=folder/'checkpoint.pt';temp_alias=folder/f'checkpoint.alias.{os.getpid()}';temp_alias.symlink_to(replacement.name)
    # A legacy checkpoint.pt is replaced only after verification and COMMITTED.
    temp_alias.replace(alias)
    if path!=alias:
        assert path.resolve().is_relative_to(root) and path!=replacement;path.unlink()
    row={'folder':str(folder),'old_commit':old,'new_commit':new,'time':time.time(),
         'all_non_optimizer_fields_bit_exact':True,'best_full_resume_preserved':str(best),
         'released_bytes':old_size-replacement.stat().st_size}
    records.append(row);atomic_json(manifest,{'state':'running','pid':os.getpid(),'records':records})
    del actual,payload;gc.collect()
    print(json.dumps({'folder':str(folder),'released_gib':row['released_bytes']/2**30,
                      'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
    if shutil.disk_usage(base).free>12*2**30:break
atomic_json(manifest,{'state':'complete','records':records,'skipped':skipped,'time':time.time(),
                      'free_gib':shutil.disk_usage(base).free/2**30})
