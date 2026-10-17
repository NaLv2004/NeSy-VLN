import sys,json,time
from pathlib import Path
root=Path('/home/user/xjy/strips_wm_navigation_20261009');sys.path.insert(0,str(root))
from stripsnav.io import read_json,load_committed,atomic_json
for marker in sorted((root/'runs').glob('*/latest/ARCHIVED_CHECKPOINT.json')):
    record=read_json(marker)
    if not Path(record['source']).exists():break
else:raise RuntimeError('No completed cold checkpoint to verify')
started=time.time();payload=load_committed(marker.parent)
assert isinstance(payload,dict) and not Path(record['source']).exists()
result={'time':time.time(),'path':str(marker.parent),'source_sha256':record['original_sha256'],
    'loaded_keys':list(payload),'seconds':time.time()-started,'no_disk_rehydration':True,
    'qualification':'load_committed verified decompressed bytes against original COMMITTED SHA256 before torch.load.'}
atomic_json(root/'cold_checkpoint_reader_verification.json',result);print(json.dumps(result,indent=2))
