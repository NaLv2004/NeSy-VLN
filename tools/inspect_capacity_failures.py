import json
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/goal_condition_capacity_v14')
for path in root.glob('*retry1/probe.log'):
    print(json.dumps({'name':path.parent.name,'tail':path.read_text(errors='replace')[-3000:]}))
