import json,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'goal_conditioned_automaton_v55'
read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json')
if old and time.time()-old['time']<600:print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for entry in read(out/'launcher.json')['jobs']:
 folder=out/entry['arm'];s=read(folder/'supervisor.json') or {};epochs=[]
 if (folder/'epochs.jsonl').exists():epochs=[json.loads(x) for x in (folder/'epochs.jsonl').read_text().splitlines()[-3:]]
 jobs[entry['arm']]={'launcher':entry,'supervisor':s,'supervisor_live':live(entry['supervisor_pid']),
  'trainer_live':live(s.get('pid',0)),'status':read(folder/'status.json'),'latest_commit':read(folder/'latest/COMMITTED'),
  'epochs_tail':epochs,'best_score':read(folder/'best_score.json'),
  'tail':(folder/'driver.log').read_text(errors='replace')[-1400:] if (folder/'driver.log').exists() else ''}
r={'time':time.time(),'jobs':jobs,'free_gib':shutil.disk_usage(b).free/2**30,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
(out/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

