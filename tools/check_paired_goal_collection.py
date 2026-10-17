import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/data/paired_goals_v12')
record=json.loads((out/'process.json').read_text());p=Path(f"/proc/{record['pid']}/cmdline")
result={'time':time.time(),'process':record,'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
    'progress':json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else None,
    'complete':json.loads((out/'COMPLETE').read_text()) if (out/'COMPLETE').exists() else None,
    'tail':(out/'collector.log').read_text(errors='replace')[-1200:]}
(out/'last_inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
