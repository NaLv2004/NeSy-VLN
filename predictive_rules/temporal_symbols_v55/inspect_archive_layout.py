import json,os
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
report={}
for name in ('run_v1','run_memory_v2','run_boundary_balanced_v43'):
    folder=ROOT/name;items=[]
    for p in folder.iterdir():
        if p.is_file():items.append(dict(name=p.name,bytes=p.stat().st_size))
        elif p.is_dir():items.append(dict(name=p.name,bytes=sum(f.stat().st_size for f in p.rglob('*') if f.is_file()),sample=[str(f.relative_to(p)) for f in list(p.rglob('*.npz'))[:2]]))
    report[name]=sorted(items,key=lambda x:-x['bytes'])[:18]
print(json.dumps(report))
