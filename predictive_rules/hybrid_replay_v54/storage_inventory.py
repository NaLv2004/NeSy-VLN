import json,shutil
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
runs=[]
for folder in ROOT.glob('run_*'):
    points=sorted(folder.glob('checkpoint_*.pt'))
    if len(points)<=3:continue
    sv=folder/'supervisor.json'
    runs.append(dict(name=folder.name,files=len(points),bytes=sum(p.stat().st_size for p in points),
        supervisor=json.loads(sv.read_text()) if sv.exists() else None,
        files_info=[dict(name=p.name,bytes=p.stat().st_size) for p in points]))
print(json.dumps(dict(free=shutil.disk_usage(ROOT).free,runs=sorted(runs,key=lambda r:-r['bytes']))))
