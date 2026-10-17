"""Archive immutable older evaluation traces, preserving checkpoints and summaries.

Training only reads summaries after a suite completes. Each archive is verified
against source hashes and its summary is checked unchanged before source removal.
"""
import hashlib,json,os,shutil,tarfile,time,gzip
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
archive_root=base/'evidence_archives/completed_evaluations';archive_root.mkdir(exist_ok=True)
runs=['symbolic_robust_v7/runs/robust_reference','symbolic_robust_v7/runs/robust_multieffect',
      'symbolic_vision_v8/runs/frozen_convnext','symbolic_vision_v8/runs/frozen_qwen',
      'symbolic_joint_v9/runs/joint','calibrated_vision_v10/runs/uncalibrated_control',
      'calibrated_vision_v10/runs/calibrated_qwen']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for relative in runs:
    run=(base/relative).resolve();assert run.is_relative_to(base)
    current=json.loads((run/'status.json').read_text())['update']
    for step in sorted((run/'evaluations').glob('step_*')):
        if int(step.name.split('_')[-1])>=current:continue
        for suite in ['development','fixed80']:
            folder=step/suite;summary=folder/'summary.json'
            if not summary.exists():continue
            sources=[p for p in folder.iterdir() if p.is_dir()]
            if not sources:continue
            name=relative.replace('/','_')+'_'+step.name+'_'+suite
            dest=archive_root/(name+'.tar.gz');manifest=archive_root/(name+'.manifest.json')
            assert not dest.exists() and not manifest.exists()
            stamp=digest(summary);files={}
            for source in sources:
                assert source.resolve().is_relative_to(run) and not source.is_symlink()
                for p in source.rglob('*'):
                    if p.is_file():
                        assert not p.is_symlink();files[str(p.relative_to(base))]={'bytes':p.stat().st_size,'sha256':digest(p)}
            temporary=dest.with_suffix('.partial')
            with temporary.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1) as packed:
                with tarfile.open(fileobj=packed,mode='w|') as archive:
                    for source in sources:archive.add(source,arcname=str(source.relative_to(base)))
            seen=set()
            with tarfile.open(temporary,'r:gz') as archive:
                for member in archive:
                    if member.isfile():
                        data=archive.extractfile(member).read();expected=files[member.name]
                        assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'];seen.add(member.name)
            assert seen==files.keys() and digest(summary)==stamp
            assert all((base/name).stat().st_size==info['bytes'] and digest(base/name)==info['sha256'] for name,info in files.items())
            temporary.replace(dest)
            record={'time':time.time(),'suite':str(folder),'archive':str(dest),'summary_sha256':stamp,'files':files,
                    'source_bytes':sum(x['bytes'] for x in files.values()),'archive_bytes':dest.stat().st_size,
                    'restore_root':str(base),'checkpoint_and_summary_preserved':True}
            manifest.write_text(json.dumps(record,indent=2))
            for source in sources:
                assert source.resolve().is_relative_to(run) and source!=run;shutil.rmtree(source)
            (folder/'ARCHIVED_TRACES.json').write_text(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2))
            print(json.dumps({k:v for k,v in record.items() if k!='files'}),flush=True)
print(json.dumps({'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
