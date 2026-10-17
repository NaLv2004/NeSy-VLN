import json,os,shutil
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve()
retired=json.loads((base/'degraded_runs_retirement.json').read_text());result=[]
for row in retired:
    run=Path(row['run']);assert run.is_relative_to(base) and (run/'STOP').exists()
    for marker in run.rglob('COMMITTED'):
        record=json.loads(marker.read_text());p=marker.parent/record['checkpoint_file']
        result.append({'folder':str(marker.parent),'file':str(p),'bytes':p.stat().st_size,
                       'links':p.stat().st_nlink,'optimizer':record.get('optimizer_included'),
                       'epoch':record.get('epoch')})
print(json.dumps({'free_gib':shutil.disk_usage(base).free/2**30,'checkpoints':result},indent=2))
(base/'retired_storage_inventory.json').write_text(json.dumps(result,indent=2))
