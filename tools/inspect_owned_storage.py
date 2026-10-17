import json,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010')
result={'disk':dict(zip(['total','used','free'],shutil.disk_usage(root))),
    'owned_folders':subprocess.check_output(['du','-x','-B1','--max-depth=3',str(root)],text=True).splitlines()}
result['owned_folders']=sorted(result['owned_folders'],key=lambda s:int(s.split()[0]),reverse=True)[:50]
result['checkpoints']=[]
for p in root.glob('*/runs/*/step_*/*'):
    if p.is_file() and p.suffix=='.pt':
        result['checkpoints'].append({'path':str(p),'bytes':p.stat().st_size,'links':p.stat().st_nlink})
out=root/'storage_inspection.json';out.write_text(json.dumps(result,indent=2));print(out.read_text())
