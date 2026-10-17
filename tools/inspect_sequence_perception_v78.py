import json,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/sequence_perception_v78')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
last=read(b/'inspection.json')
if last and time.time()-last['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if pid and p.exists() else None
jobs={}
for j in read(b/'launcher.json')['jobs']:
 folder=b/j['arm'];supervisor=read(folder/'supervisor.json');status=read(folder/'status.json')
 x=dict(j,supervisor=supervisor,supervisor_live=live(j['supervisor_pid']),trainer_live=live((supervisor or {}).get('pid')),
        status=status,commit=read(folder/'latest/COMMITTED'),best=read(folder/'best_score.json'),
        evaluations={p.parent.name:read(p) for p in sorted((folder/'evaluations').glob('*/summary.json'))})
 if not status or status.get('state')=='failed':x['tail']=(folder/'driver.log').read_text(errors='replace')[-5000:]
 jobs[j['arm']]=x
r={'time':time.time(),'jobs':jobs,'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
(b/'inspection.json').write_text(json.dumps(r,indent=2))
for x in r['jobs'].values():
 if x['status']:x['status']={k:v for k,v in x['status'].items() if k not in ['paired_goal_audit']}
 x['evaluations']={k:{a:v[a] for a in ['episodes','successes','success_rate','collision_action_rate']} for k,v in x['evaluations'].items()}
print(json.dumps(r,indent=2))
