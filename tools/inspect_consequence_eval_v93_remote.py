import json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/consequence_refinement_v93')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
last=read(b/'evaluation_inspection.json')
if last and time.time()-last['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
process=read(b/'evaluation_process.json');path=Path(f'/proc/{process["pid"]}/cmdline')
r=dict(time=time.time(),process=process,live=path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else None,
 status=read(b/'development/status.json'),summary=read(b/'development/summary.json'),
 subprocesses=read(b/'development/processes.json'))
if not r['live'] or (r['status'] or {}).get('state')=='failed':r['log_tail']=(b/'evaluation.log').read_text(errors='replace')[-5500:]
(b/'evaluation_inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
