"""Retain cold model weights exactly; discard unused historical optimizer slots."""
import gc,json,math,os,shutil,sys,time
from pathlib import Path
import torch,numpy as np
workspace=Path('/home/user/xjy/nesy_vla_20261010').resolve()
classifiers='--retired-classifiers' in sys.argv
base=Path('/home/user/xjy/strips_wm_navigation_20261009/runs').resolve() if classifiers else workspace
sys.path.insert(0,'/home/user/xjy/strips_wm_navigation_20261009')
from stripsnav.io import load_committed,read_json,atomic_json,sha256
allowed={'symbolic_execution_v38'}
if classifiers:allowed={f'strict_symbolic_v1_wm{i}' for i in range(6)}|{'wm_nav_outcomes_b7','pilot_nav_b5_e25'}
audit=workspace/('obsolete_classifier_optimizer_compaction_v34.json' if classifiers else 'redundant_best_optimizer_compaction_v61.json')
records=read_json(audit).get('records',[]) if audit.exists() else []
def live(folder):
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name==str(os.getpid()):continue
        try:
            if str(folder).encode() in path.read_bytes():return True
        except OSError:pass
    return False
def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and math.isnan(a):return isinstance(b,float) and math.isnan(b)
    return a==b
for name in sorted(allowed):
    root=base/name
    if not root.exists():continue
    for marker in sorted(root.rglob('COMMITTED')):
        folder=marker.parent;run=folder.parent
        if folder.name!='best_development':continue
        latest=run/'latest/COMMITTED'
        if not latest.exists():continue
        latest_commit=read_json(latest);best_commit=read_json(marker)
        if latest_commit.get('epoch')!=best_commit.get('epoch'):continue
        if not latest_commit.get('optimizer_included'):continue
        assert sha256(latest.parent/latest_commit['checkpoint_file'])==latest_commit['checkpoint_sha256']
        status=read_json(run/'status.json')
        if status.get('state')!='complete':continue
        if classifiers and (folder.name!='best' or folder.parent.name!='visual_classifier'):continue
        # v12 remains a live source tree for probes; only its completed, separate integration run qualifies.
        if name=='symbolic_paired_v12' and 'integration' not in folder.relative_to(root).parts:continue
        if 'initialization_source' in folder.parts or live(run):continue
        old=read_json(marker)
        if old.get('optimizer_included') is False or 'checkpoint_file' not in old:continue
        path=folder/old['checkpoint_file']
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink!=1:continue
        assert path.resolve().is_relative_to(root) and sha256(path)==old['checkpoint_sha256']
        payload=load_committed(folder)
        if not isinstance(payload.get('optimizer'),dict):continue
        old_bytes=path.stat().st_size;payload['optimizer']=None
        full=load_committed(run/'latest')
        assert same(payload['model'],full['model']), 'Best/latest model mismatch'
        del full
        temporary=folder/f'optimizer_pruned.{time.time_ns()}.tmp'
        with temporary.open('wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        restored=torch.load(temporary,map_location='cpu',weights_only=False)
        assert same(payload,restored) and read_json(marker)==old and not live(run)
        replacement=temporary.with_suffix('.pt');temporary.replace(replacement)
        metadata={k:v for k,v in old.items() if k not in ['checkpoint_file','checkpoint_sha256','full_optimizer_available_at']}
        metadata.update(optimizer_included=False,optimizer_pruned=True,original_full_checkpoint_sha256=old['checkpoint_sha256'],
            pruning_reason='Obsolete training experiment; every model tensor and all non-optimizer fields retained exactly.')
        metadata['full_optimizer_available_at']=str(run/'latest')
        new=dict(metadata,checkpoint_file=replacement.name,checkpoint_sha256=sha256(replacement))
        atomic_json(marker,new);atomic_json(folder/'metadata.json',metadata)
        alias=folder/'checkpoint.pt';temporary_alias=folder/f'.alias.{os.getpid()}';temporary_alias.symlink_to(replacement.name);temporary_alias.replace(alias)
        if path!=alias:path.unlink()
        records.append({'folder':str(folder),'old_commit':old,'new_commit':new,'released_bytes':old_bytes-replacement.stat().st_size,
            'all_non_optimizer_content_bit_exact':True,'time':time.time()})
        atomic_json(audit,{'state':'running','pid':os.getpid(),'records':records})
        print(json.dumps({'folder':str(folder),'freed_gib':records[-1]['released_bytes']/2**30,
            'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
        del restored,payload;gc.collect()
        if shutil.disk_usage(base).free>=8*2**30:break
    if shutil.disk_usage(base).free>=8*2**30:break
atomic_json(audit,{'state':'complete','records':records,'time':time.time(),'free_gib':shutil.disk_usage(base).free/2**30,
    'protected_families':['symbolic_robust_v7','symbolic_conjunctive_v23','symbolic_recovery_v28','symbolic_execution_v38','token_revision_rl_v45'],
    'all_model_weights_and_training_validation_evidence_retained':True})
