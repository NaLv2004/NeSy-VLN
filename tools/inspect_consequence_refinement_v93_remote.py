import json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/consequence_refinement_v93')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
last=read(b/'inspection.json')
if last and time.time()-last['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
process=read(b/'process.json');p=Path(f'/proc/{process["pid"]}/cmdline')
r=dict(time=time.time(),process=process,live=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
 status=read(b/'status.json'),baseline=read(b/'baseline.json'),complete=read(b/'COMPLETE'),
 initialization=read(b/'initialization_verification.json'),probes={x.name:read(x) for x in b.glob('probe_*.json')})
if not r['live'] or (r['status'] or {}).get('state')=='failed':r['log_tail']=(b/'driver.log').read_text(errors='replace')[-6500:]
(b/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
