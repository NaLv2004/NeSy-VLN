import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/critic_isolation_rl_v50')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json')
if old and time.time()-old['time']<600:print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for entry in read(out/'launcher.json')['jobs']:
 folder=out/entry['arm'];s=read(folder/'supervisor.json') or {}
 jobs[entry['arm']]={'supervisor':s,'supervisor_live':live(entry['supervisor_pid']),'trainer_live':live(s.get('pid',0)),
 'status':read(folder/'status.json'),'commit':read(folder/'latest/COMMITTED'),'best':read(folder/'best_score.json'),
 'tail':(folder/'driver.log').read_text(errors='replace')[-1200:] if (folder/'driver.log').exists() else ''}
r={'time':time.time(),'jobs':jobs};(out/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
