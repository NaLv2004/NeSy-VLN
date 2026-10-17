import hashlib,json,subprocess,tempfile,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');archive=base/'evidence_archives'
index=next((archive/'completed_evaluations').glob('*.tar.gz.content.json'))
record=json.loads(index.read_text());member=next(m for m in record['members'] if m['path'].endswith('.png'))
with tempfile.TemporaryDirectory(prefix='restore_check_',dir=archive) as temp:
    target=Path(temp).resolve();assert target.is_relative_to(archive)
    result=subprocess.run(['python3',str(archive/'restore_cold_evidence.py'),str(index),'--destination',str(target),
        '--suffix',member['path']],capture_output=True,text=True,check=True)
    info=json.loads(result.stdout);assert info['restored_files']==1
    data=(target/member['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==member['sha256']
out={'time':time.time(),'index':str(index),'file':member['path'],'sha256':member['sha256'],
    'restore_tool_executed':True,'exact_original_image_bytes_restored':True,'temporary_copy_removed':True}
(archive/'restore_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
# Keep descriptive metadata aligned for historical files restored to an already shared full checkpoint.
manifest=Path('/home/user/xjy/strips_wm_navigation_20261009/nested_optimizer_compaction.json')
for row in json.loads(manifest.read_text())['records']:
    if not row.get('reverted_to_shared_full_checkpoint'):continue
    folder=Path(row['folder']);old=row['old_commit'];assert json.loads((folder/'COMMITTED').read_text())==old
    (folder/'metadata.json').write_text(json.dumps({k:v for k,v in old.items() if k not in ['checkpoint_file','checkpoint_sha256']},indent=2))
