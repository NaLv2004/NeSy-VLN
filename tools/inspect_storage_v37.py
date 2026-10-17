import json,os,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');rows=[]
for name in ['symbolic_teacher_v25','symbolic_task_v15','symbolic_conjunctive_v23','symbolic_branch_v17']:
    root=base/name
    for p in root.rglob('COMMITTED'):
        m=json.loads(p.read_text());path=p.parent/m['checkpoint_file']
        if path.exists():rows.append({'folder':str(p.parent),'gib':path.stat().st_size/2**30,'links':path.stat().st_nlink,'optimizer':m.get('optimizer_included'),'file':str(path)})
result={'time':time.time(),'free_gib':shutil.disk_usage(base).free/2**30,'checkpoints':rows}
(base/'storage_inventory_v37.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
