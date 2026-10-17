import json
from pathlib import Path
R=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out={}
for name in ('run_v1','run_memory_v2','run_boundary_balanced_v43'):
    p=R/name;rows=[]
    for d in sorted([*p.glob('development_step_*'),*p.glob('dev_*')]):
        summary=d/'summary.json'
        data=json.loads(summary.read_text()) if summary.exists() else {}
        rows.append(dict(name=d.name,episodes=data.get('episodes'),successes=data.get('successes'),keys=list(data)[:10],
            bytes=sum(x.stat().st_size for x in d.glob('*/program_*.npz'))))
    out[name]=rows
print(json.dumps(out))
