import json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/text_context_v70');read=lambda p:json.loads(p.read_text()) if p.exists() else None
previous=read(b/'inspection.json')
if previous and time.time()-previous['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':previous['time']+600}));raise SystemExit
m=read(b/'launcher.json');p=Path(f'/proc/{m["pid"]}/cmdline');s=read(b/'status.json')
r={'time':time.time(),'launcher':m,'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
 'status':s,'result':read(b/'result.json')}
if not s or s.get('state')=='failed':r['tail']=(b/'driver.log').read_text(errors='replace')[-2500:]
(b/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
