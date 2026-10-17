import json,time
from pathlib import Path
folder=Path('/home/user/xjy/nesy_vla_20261010/evidence_archives')
meta=json.loads((folder/'active_maintenance_v38_process.json').read_text());proc=Path(f'/proc/{meta["pid"]}/cmdline')
status=folder/'active_maintenance_v38.json'
result={'time':time.time(),'meta':meta,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
    'status':json.loads(status.read_text()) if status.exists() else None,
    'tail':(folder/'maintenance_launcher_v38.log').read_text(errors='replace')[-1000:]}
(folder/'active_maintenance_v38_startup.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
