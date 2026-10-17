import json
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_induced_effects_v39')
path=sorted(root.glob('reachability_audit_*.json'))[-1]
data=json.loads(path.read_text())
print(json.dumps(dict(path=str(path),checkpoint=data['checkpoint'],summary=data['summary'],seconds=data['seconds'])))
