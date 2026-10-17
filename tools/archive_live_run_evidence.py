"""Losslessly archive immutable traces directly, without a temporary tar copy."""
import hashlib,json,os,shutil,time,zlib
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve();store=base/'evidence_archives/content_store'
indices=base/'evidence_archives/direct_indices';indices.mkdir(parents=True,exist_ok=True)
report=base/'evidence_archives/direct_archive_status.json';records=[]
runs=['symbolic_robust_v7/runs/robust_multieffect','symbolic_joint_v9/runs/joint',
    'symbolic_vision_v8/runs/frozen_convnext','symbolic_vision_v8/runs/frozen_qwen',
    'symbolic_reward_v11/runs/original','symbolic_reward_v11/runs/bounded_repeat_stop40',
    'symbolic_task_v15/runs/paired_tasks','symbolic_branch_v17/runs/branch_current',
    'symbolic_conjunctive_v23/runs/branch_control','symbolic_conjunctive_v23/runs/conjunctive',
    'symbolic_teacher_v25/runs/ema_control','symbolic_teacher_v25/runs/synchronous',
    'symbolic_recovery_v28/runs/base_control','symbolic_recovery_v28/runs/recovery_mixture']
def atomic(path,value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
def digest(data):return hashlib.sha256(data).hexdigest()
for relative in runs:
    run=(base/relative).resolve()
    if not (run/'status.json').exists():continue
    status=json.loads((run/'status.json').read_text());current=status['update'];targets=[]
    retired=(run/'STOP').exists() or status.get('state')=='complete'
    for step in sorted((run/'evaluations').glob('step_*')):
        # summary.json is published only after every rank has finished writing
        # this suite and crossed the evaluation barrier. Its trace bytes are
        # immutable even when checkpoint storage is blocking the next update.
        for suite in ['development','fixed80']:
            folder=step/suite
            if (folder/'summary.json').exists():targets.append(folder)
    for branch in ['traces','demonstration_traces']:
        for update in sorted((run/branch).glob('update_*')):
            if int(update.name.split('_')[-1])<current-10:targets.append(update)
    for folder in targets:
        folder=folder.resolve();assert folder.is_relative_to(run) and folder!=run
        name=str(folder.relative_to(base)).replace('/','_');index=indices/(name+'.content.json')
        # Existing conventional archives have already removed their trace directories.
        sources=[p for p in folder.iterdir() if p.name!='ARCHIVED_TRACES.json' and
            (p.is_dir() or (folder.parent.name=='demonstration_traces' and p.is_file()))]
        if not sources:continue
        if index.exists():record=json.loads(index.read_text())
        else:
            members=[];new_bytes=0
            for source in sources:
                assert source.resolve().is_relative_to(folder) and not source.is_symlink()
                for path in ([source] if source.is_file() else sorted(source.rglob('*'))):
                    if not path.is_file():continue
                    assert not path.is_symlink();content=path.read_bytes();sha=digest(content)
                    blob=store/sha[:2]/(sha+'.zlib');blob.parent.mkdir(exist_ok=True)
                    if not blob.exists():
                        payload=zlib.compress(content,1);tmp=blob.with_name(blob.name+f'.{os.getpid()}.tmp')
                        tmp.write_bytes(payload);tmp.replace(blob);new_bytes+=len(payload)
                    assert digest(zlib.decompress(blob.read_bytes()))==sha
                    members.append({'path':str(path.relative_to(base)),'bytes':len(content),'sha256':sha,'blob':str(blob)})
            record={'time':time.time(),'source':str(folder),'members':members,'new_blob_bytes':new_bytes,
                'original_bytes':sum(m['bytes'] for m in members),'restore_root':str(base),
                'all_original_files_byte_verified':True,'restore_tool':'restore_cold_evidence.py'}
            atomic(index,record)
        for member in record['members']:
            assert digest(zlib.decompress(Path(member['blob']).read_bytes()))==member['sha256']
            path=base/member['path'];assert path.resolve().is_relative_to(folder)
            if path.exists():assert digest(path.read_bytes())==member['sha256']
        atomic(folder/'ARCHIVED_TRACES.json',{'content_index':str(index),'archive_format':'lossless_deduplicated_files','restore_root':str(base)})
        for source in sources:
            assert source.resolve().is_relative_to(folder) and source!=folder
            if source.is_dir():shutil.rmtree(source)
            else:source.unlink()
        row={k:v for k,v in record.items() if k!='members'};records.append(row)
        atomic(report,{'state':'running','pid':os.getpid(),'time':time.time(),'records':records,'free_gib':shutil.disk_usage(base).free/2**30})
        print(json.dumps({'source':str(folder),'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
atomic(report,{'state':'complete','pid':os.getpid(),'time':time.time(),'records':records,'free_gib':shutil.disk_usage(base).free/2**30})
