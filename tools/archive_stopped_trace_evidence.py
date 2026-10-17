"""Losslessly archive only bulky trace evidence of explicitly stopped old runs.

Model checkpoints, validation summaries, source code and fixed split files keep
their existing paths. Every archived byte is verified before removing sources.
"""
import argparse,gzip,hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path

base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
archive_root=base/'evidence_archives';archive_root.mkdir(exist_ok=True)
active=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',default=['runs/full_v1','v2/runs/full_v2','v3/runs/full_v3']);args=parser.parse_args()
outcomes=[]
for relative in args.runs:
    run=(base/relative).resolve()
    assert run.is_relative_to(base) and run!=base
    if any(str(run) in command and 'scripts/train.py' in command for command in active):
        raise RuntimeError('Refusing to archive a live run: '+str(run))
    sources=[p for p in [run/'traces',run/'demonstration_traces'] if p.is_dir()]
    for suite in (run/'evaluations').glob('step_*/*'):
        if suite.name not in ['development','fixed80'] or not (suite/'summary.json').exists():continue
        sources.extend(p for p in suite.iterdir() if p.is_dir())
    sources=[p.resolve() for p in sources]
    assert all(p.is_relative_to(run) for p in sources)
    if not sources:continue
    name=relative.replace('/','_');destination=archive_root/(name+'.tar.gz')
    manifest=archive_root/(name+'.manifest.json')
    if destination.exists() or manifest.exists():raise RuntimeError('Archive exists; will not replace evidence')
    files={}
    for source in sources:
        for p in source.rglob('*'):
            if not p.is_file():continue
            assert not p.is_symlink()
            content=p.read_bytes();files[str(p.relative_to(base))]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    temporary=destination.with_suffix('.partial')
    with temporary.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1) as packed:
        with tarfile.open(fileobj=packed,mode='w|') as archive:
            for source in sources:archive.add(source,arcname=str(source.relative_to(base)),recursive=True)
    verified=set()
    with tarfile.open(temporary,'r:gz') as archive:
        for member in archive:
            if not member.isfile():continue
            expected=files[member.name];content=archive.extractfile(member).read()
            assert len(content)==expected['bytes'] and hashlib.sha256(content).hexdigest()==expected['sha256']
            verified.add(member.name)
    assert verified==files.keys()
    temporary.replace(destination)
    data={'time':time.time(),'run':str(run),'archive':str(destination),'files':files,
        'source_bytes':sum(v['bytes'] for v in files.values()),'archive_bytes':destination.stat().st_size,
        'restore':'Extract this archive under /home/user/xjy/nesy_vla_20261010 to restore original paths.',
        'checkpoints_and_summaries':'All training checkpoints and evaluation summaries remain at original paths.'}
    manifest.write_text(json.dumps(data,indent=2))
    for source in sources:
        assert source.is_relative_to(run) and source!=run and source.is_dir()
        shutil.rmtree(source)
    compact={k:v for k,v in data.items() if k!='files'};outcomes.append(compact)
    print(json.dumps(compact),flush=True)
summary=archive_root/'summary.json';previous=json.loads(summary.read_text()) if summary.exists() else []
summary.write_text(json.dumps(previous+outcomes,indent=2))
print(json.dumps({'free_bytes':shutil.disk_usage(base).free}),flush=True)
