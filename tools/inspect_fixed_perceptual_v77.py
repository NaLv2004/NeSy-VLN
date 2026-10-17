import json,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010/fixed_perceptual_v77')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
last=read(B/'inspection.json')
if last and time.time()-last['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
supervisor=read(B/'supervisor.json')
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if pid and p.exists() else None
r={'time':time.time(),'supervisor':supervisor,'supervisor_live':live((supervisor or {}).get('supervisor_pid')),
 'trainer_live':live((supervisor or {}).get('pid')),'status':read(B/'fixed80/status.json'),'summary':read(B/'fixed80/summary.json')}
if (supervisor or {}).get('state')=='failed':r['error_tail']=(B/'driver.log').read_text()[-5000:]
(B/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
