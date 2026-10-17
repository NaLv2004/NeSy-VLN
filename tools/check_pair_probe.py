import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/paired_goal_probe_v12')
record=json.loads((out/'probe_process.json').read_text());p=Path(f"/proc/{record['pid']}/cmdline")
result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
item={'time':time.time(),'process':record,'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
    'result_state':(result or {}).get('state'),'initial':(result or {}).get('initial'),
    'latest_learning':((result or {}).get('learning') or [None])[-1],
    'tail':(out/'probe.log').read_text(errors='replace')[-1500:]}
(out/'last_inspection.json').write_text(json.dumps(item,indent=2))
def compact(value):
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items() if k not in ['labels','predictions']}
    return value
print(json.dumps(compact(item),indent=2))
