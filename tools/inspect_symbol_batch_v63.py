import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/numerical_symbol_batch_v63')
meta=json.loads((out/'launcher.json').read_text());p=Path(f'/proc/{meta["pid"]}/cmdline')
result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
r={'time':time.time(),'launcher':meta,'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,'result':result,'tail':(out/'driver.log').read_text()[-1000:]}
(out/'inspection.json').write_text(json.dumps(r,indent=2));r.pop('tail') if result else None;print(json.dumps(r,indent=2))
