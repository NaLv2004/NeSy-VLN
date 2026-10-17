import json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'frozen_rule_learning_v39';read=lambda p:json.loads(p.read_text()) if p.exists() else None
saved=read(out/'inspection.json')
if saved and time.time()-saved['time']<600:
    print(json.dumps({'skipped':'minimum600seconds','next_check':saved['time']+600}));raise SystemExit
launch=read(out/'launcher.json');proc=Path(f'/proc/{launch["pid"]}/cmdline');r=read(out/'result.json')
result={'time':time.time(),'launcher':launch,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
    'result':r,'tail':(out/'driver.log').read_text(errors='replace')[-1200:]}
(out/'inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
