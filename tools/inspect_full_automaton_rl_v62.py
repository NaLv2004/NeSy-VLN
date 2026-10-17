import json,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'full_automaton_rl_v62'
read=lambda p:json.loads(p.read_text()) if p.exists() else None
previous=read(out/'inspection.json')
startup=read(out/'startup_inspection.json') or {}
last_check=max((previous or {}).get('time',0),startup.get('time',0))
if time.time()-last_check<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last_check+600}));raise SystemExit
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for entry in read(out/'launcher.json')['jobs']:
 folder=out/entry['arm'];s=read(folder/'supervisor.json') or {}
 jobs[entry['arm']]={'launcher':entry,'supervisor':s,'supervisor_live':live(entry['supervisor_pid']),
  'trainer_live':live(s.get('pid',0)),'status':read(folder/'status.json'),'latest_commit':read(folder/'latest/COMMITTED'),
  'best_score':read(folder/'best_score.json'),'startup':read(folder/'startup_verified.json'),
  'tail':(folder/'driver.log').read_text(errors='replace')[-2000:] if (folder/'driver.log').exists() else ''}
r={'time':time.time(),'jobs':jobs,'free_gib':shutil.disk_usage(b).free/2**30,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
(out/'inspection.json').write_text(json.dumps(r,indent=2))
for j in r['jobs'].values():
 for field in ['status','best_score']:
  d=j.get(field) or {}
  for name in ['development','training']:
   if isinstance(d.get(name),dict):d[name]={k:v for k,v in d[name].items() if k!='groups'}
  d.pop('groups',None)
 if (j.get('status') or {}).get('state') not in (None,'failed'):j.pop('tail',None)
 j.pop('startup',None)
print(json.dumps(r,indent=2))
