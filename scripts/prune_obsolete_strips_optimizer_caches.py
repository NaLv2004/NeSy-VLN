"""Drop unused optimizer caches from stopped pre-NeSy trials, preserving models.

The user authorized removing useless old training checkpoints. This retains
every non-optimizer field and explicitly records that these legacy checkpoints
are now evaluation/weight-initialization snapshots, not exact optimizer resumes.
"""
import gc,json,math,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
root=Path('/home/user/xjy/strips_wm_navigation_20261009').resolve();base=Path('/home/user/xjy/nesy_vla_20261010')
sys.path.insert(0,str(root));from stripsnav.io import read_json,load_committed,atomic_json,sha256
prefixes=('strict_symbolic_v1_wm','discrete_inverse_restore_v1_warm','wm_nav_b','wm_nav_pretrained_b',
          'wm_nav_pretrained_discrete_inverse_b','pilot_nav_')
manifest=base/'obsolete_strips_optimizer_cache_pruning.json';records=[]
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
for run in sorted((root/'runs').iterdir()):
    if not run.is_dir() or not run.name.startswith(prefixes) or live(run):continue
    folder=run/'best';marker=folder/'COMMITTED'
    if not marker.exists():continue
    old=read_json(marker);path=folder/old.get('checkpoint_file','checkpoint.pt')
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink!=1:continue
    payload=load_committed(folder)
    if not isinstance(payload.get('optimizer'),dict):del payload;continue
    payload['optimizer']=None;size=path.stat().st_size
    assert path.resolve().is_relative_to(root/'runs') and not live(run)
    newfile=folder/f'checkpoint.optimizer_pruned.{time.time_ns()}.pt';temporary=newfile.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    actual=torch.load(temporary,map_location='cpu',weights_only=False)
    assert same(payload,actual) and read_json(marker)==old and not live(run)
    temporary.replace(newfile)
    metadata={k:v for k,v in old.items() if k not in ['checkpoint_file','checkpoint_sha256']}
    metadata.update(optimizer_included=False,optimizer_pruned=True,exact_optimizer_resume_available=False,
        original_full_checkpoint_sha256=old['checkpoint_sha256'])
    new=dict(metadata,checkpoint_file=newfile.name,checkpoint_sha256=sha256(newfile))
    atomic_json(marker,new);atomic_json(folder/'metadata.json',metadata)
    alias=folder/'checkpoint.pt';tmpalias=folder/f'checkpoint.alias.{os.getpid()}';tmpalias.symlink_to(newfile.name);tmpalias.replace(alias)
    if path!=alias:path.unlink()
    row={'folder':str(folder),'old_commit':old,'new_commit':new,'all_non_optimizer_fields_bit_exact':True,
        'released_bytes':size-newfile.stat().st_size};records.append(row)
    atomic_json(manifest,{'state':'running','pid':os.getpid(),'records':records})
    del payload,actual;gc.collect()
    print(json.dumps({'folder':str(folder),'released_gib':row['released_bytes']/2**30,'free_gib':shutil.disk_usage(root).free/2**30}),flush=True)
    if shutil.disk_usage(root).free>9*2**30:break
atomic_json(manifest,{'state':'complete','time':time.time(),'records':records,'free_gib':shutil.disk_usage(root).free/2**30})
