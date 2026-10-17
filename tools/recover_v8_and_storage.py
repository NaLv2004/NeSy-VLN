import json,time,shutil
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_vision_v8'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def proc(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
out={'time':time.time(),'free_gib':shutil.disk_usage(base).free/2**30,'controller':read(root/'controller.json'),'held':read(root/'controller_held.json'),'allocation':read(root/'allocation.json')}
out['controller_live']=proc(out['controller']['pid'])
out['new_supervisors']={str(p):read(p) for p in (root/'runs').glob('*/supervisor.json')}
out['archives']={}
for name in ['v5_plain','v5_spatial']:
    p=base/'evidence_archives'/f'{name}_controller.json';record=read(p)
    logs=base/'evidence_archives'/f'{name}_archive.log'
    out['archives'][name]={'controller':record,'live':proc((record or {}).get('pid')),'tail':logs.read_text(errors='replace')[-1800:] if logs.exists() else None}
print(json.dumps(out,indent=2))
