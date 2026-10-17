import json,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'probabilistic_automaton_v56'
read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json')
if old and time.time()-old['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for entry in read(out/'launcher.json')['jobs']:
 folder=out/entry['arm'];s=read(folder/'supervisor.json') or {};epochs=[]
 if (folder/'epochs.jsonl').exists():epochs=[json.loads(x) for x in (folder/'epochs.jsonl').read_text().splitlines()[-2:]]
 jobs[entry['arm']]={'launcher':entry,'supervisor':s,'supervisor_live':live(entry['supervisor_pid']),
 'trainer_live':live(s.get('pid',0)),'status':read(folder/'status.json'),'latest_commit':read(folder/'latest/COMMITTED'),
 'epochs_tail':epochs,'best_score':read(folder/'best_score.json'),
 'tail':(folder/'driver.log').read_text(errors='replace')[-1200:] if (folder/'driver.log').exists() else ''}
r={'time':time.time(),'jobs':jobs,'free_gib':shutil.disk_usage(b).free/2**30,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
(out/'inspection.json').write_text(json.dumps(r,indent=2))
compact=dict(time=r['time'],free_gib=r['free_gib'],gpu=r['gpu'],jobs={})
for name,j in jobs.items():
 s=j['status'] or {};best=j['best_score'] or {};dev=s.get('development') or {}
 compact['jobs'][name]={k:j[k] for k in ['launcher','supervisor','supervisor_live','trainer_live','latest_commit']}
 compact['jobs'][name].update(status={k:v for k,v in s.items() if k not in ['development','groups']},
  development={k:v for k,v in dev.items() if k!='groups'},best={k:v for k,v in best.items() if k!='groups'})
 if s.get('state')=='failed':compact['jobs'][name]['tail']=j['tail']
print(json.dumps(compact,indent=2))
