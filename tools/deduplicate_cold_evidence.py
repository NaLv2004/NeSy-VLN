"""Lossless content-addressed storage for immutable archived evidence.

Each file is verified against the previous manifest before an archive is replaced
by its file index. Images, proofs, metrics and checkpoints are all retained.
"""
import gzip,hashlib,json,os,shutil,tarfile,time,zlib
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010/evidence_archives').resolve()
root=base/'content_store';root.mkdir(exist_ok=True)
status=root/'conversion.json';records=[]
if status.exists():records=json.loads(status.read_text()).get('records',[])

def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj));tmp.replace(path)

deferred=[]
archives=list((base/'completed_evaluations').glob('*.tar.gz'))+list(base.glob('*.tar.gz'))
for archive in sorted(archives,key=lambda p:p.stat().st_size):
    index=archive.with_suffix(archive.suffix+'.content.json')
    if index.exists():continue
    if shutil.disk_usage(base).free<archive.stat().st_size+2**30:
        deferred.append(str(archive));continue
    prior=archive.with_name(archive.name.removesuffix('.tar.gz')+'.manifest.json')
    expected=json.loads(prior.read_text())['files'];members=[];seen=set();written=0;size=archive.stat().st_size
    with tarfile.open(archive,'r|gz') as source:
        for member in source:
            if member.isdir():members.append({'path':member.name,'directory':True});continue
            if not member.isfile():raise ValueError('Non-regular archived member: '+member.name)
            data=source.extractfile(member).read();digest=hashlib.sha256(data).hexdigest()
            assert digest==expected[member.name]['sha256'] and len(data)==expected[member.name]['bytes']
            blob=root/digest[:2]/(digest+'.zlib');blob.parent.mkdir(exist_ok=True)
            if not blob.exists():
                payload=zlib.compress(data,1);tmp=blob.with_suffix('.tmp');tmp.write_bytes(payload);tmp.replace(blob);written+=len(payload)
            restored=zlib.decompress(blob.read_bytes())
            assert len(restored)==len(data) and hashlib.sha256(restored).hexdigest()==digest
            members.append({'path':member.name,'bytes':len(data),'sha256':digest,'blob':str(blob)});seen.add(member.name)
    assert seen==expected.keys()
    record={'time':time.time(),'original_archive':str(archive),'original_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'original_manifest':str(prior),'all_original_files_byte_verified':True,'members':members,
        'original_archive_bytes':size,'new_blob_bytes':written,'restore_tool':'restore_cold_evidence.py'}
    atomic(index,record)
    # Re-read the committed index and validate every referenced blob before freeing the old container.
    for member in json.loads(index.read_text())['members']:
        if member.get('directory'):continue
        assert hashlib.sha256(zlib.decompress(Path(member['blob']).read_bytes())).hexdigest()==member['sha256']
    archive.unlink()
    prior_data=json.loads(prior.read_text());prior_data['content_index']=str(index);atomic(prior,prior_data)
    suite=Path(prior_data.get('suite',prior_data.get('run')));sidecar=suite/'ARCHIVED_TRACES.json'
    if sidecar.exists():
        info=json.loads(sidecar.read_text());info.update(content_index=str(index),archive_format='lossless_deduplicated_files');atomic(sidecar,info)
    records.append({k:v for k,v in record.items() if k!='members'})
    atomic(status,{'state':'running','records':records,'free_gib':shutil.disk_usage(base).free/2**30})
    print(json.dumps({'archive':archive.name,'files':len(seen),'released_gib':(size-written)/2**30,
        'free_gib':shutil.disk_usage(base).free/2**30}),flush=True)
atomic(status,{'state':'complete','records':records,'released_bytes':sum(r['original_archive_bytes']-r['new_blob_bytes'] for r in records),
    'deferred':deferred,'free_gib':shutil.disk_usage(base).free/2**30})
