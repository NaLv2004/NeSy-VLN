"""Retain exact loaded payloads, including optimizer/RNG, for retired final aliases."""
import gc,json,math,os,shutil,sys,time
from pathlib import Path
import numpy as np,torch
base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
sys.path.insert(0,'/home/user/xjy/strips_wm_navigation_20261009')
from stripsnav.io import load_committed,read_json,atomic_json,sha256

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and math.isnan(a):return isinstance(b,float) and math.isnan(b)
    return a==b

def live(run):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:
            if str(run).encode() in p.read_bytes():return True
        except OSError:pass
    return False

rows=[];audit=base/'retired_execution_coalescing_v46.json'
for family in ['symbolic_execution_v38']:
    for run in (base/family/'runs').glob('*'):
        if not (run/'status.json').exists() or read_json(run/'status.json').get('state')!='complete' or live(run):continue
        marker=read_json(run/'latest/COMMITTED');source=run/'latest'/marker['checkpoint_file']
        assert source.resolve().is_relative_to(base) and sha256(source)==marker['checkpoint_sha256']
        source_payload=load_committed(run/'latest')
        for folder in run.glob('step_*'):
            if not (folder/'COMMITTED').exists():continue
            old=read_json(folder/'COMMITTED')
            if old.get('epoch')!=marker.get('epoch'):continue
            target=folder/old['checkpoint_file']
            if source.stat().st_ino==target.stat().st_ino:continue
            assert target.resolve().is_relative_to(run) and not target.is_symlink()
            target_payload=load_committed(folder)
            if not equal(source_payload,target_payload):
                rows.append({'folder':str(folder),'skipped':'Nonidentical payload'});continue
            assert not live(run) and read_json(folder/'COMMITTED')==old
            before=target.stat();temporary=target.with_name(target.name+f'.coalesce.{os.getpid()}')
            os.link(source,temporary);temporary.replace(target)
            new=dict(old,checkpoint_sha256=marker['checkpoint_sha256'],original_serialization_sha256=old['checkpoint_sha256'],
                coalesced_full_payload_equal=True)
            atomic_json(folder/'COMMITTED',new)
            metadata=read_json(folder/'metadata.json') if (folder/'metadata.json').exists() else {}
            metadata.update(coalesced_full_payload_equal=True);atomic_json(folder/'metadata.json',metadata)
            assert sha256(target)==new['checkpoint_sha256']
            rows.append({'folder':str(folder),'source':str(source),'old_commit':old,'new_commit':new,
                'released_bytes':before.st_size if before.st_nlink==1 else 0,
                'optimizer_rng_and_all_model_tensors_equal':True})
            atomic_json(audit,{'state':'running','records':rows,'time':time.time()})
            del target_payload;gc.collect()
        del source_payload;gc.collect()
result={'state':'complete','time':time.time(),'records':rows,'free_gib':shutil.disk_usage(base).free/2**30,
    'released_gib':sum(r.get('released_bytes',0) for r in rows)/2**30}
atomic_json(audit,result);print(json.dumps(result,indent=2))
