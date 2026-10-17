import json,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'higher_order_capacity_v64'
read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json')
if old and time.time()-old['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for e in read(out/'launcher.json')['jobs']:
 f=out/e['arm'];s=read(f/'supervisor.json') or {}
 jobs[e['arm']]={'launcher':e,'supervisor':s,'supervisor_live':live(e['supervisor_pid']),'trainer_live':live(s.get('pid',0)),
  'status':read(f/'status.json'),'commit':read(f/'latest/COMMITTED'),'best_score':read(f/'best_score.json'),
  'tail':(f/'driver.log').read_text(errors='replace')[-1800:] if (f/'driver.log').exists() else ''}
r={'time':time.time(),'jobs':jobs,'free_gib':shutil.disk_usage(b).free/2**30,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
(out/'inspection.json').write_text(json.dumps(r,indent=2))
for j in jobs.values():
 for k in ['status','best_score']:
  d=j.get(k) or {};d.pop('groups',None)
  if isinstance(d.get('development'),dict):d['development'].pop('groups',None)
 if (j.get('status') or {}).get('state') not in (None,'failed'):j.pop('tail',None)
print(json.dumps(r,indent=2))
