import json,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_goal_equivalence_v57';DIAG=ROOT/'diagnostic_goal_equivalence_v57'
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
sv=read(OUT/'supervisor.json',{});ranks=[read(OUT/f'rank{r}.status.json',{}) for r in (0,1)]
handoff=read(DIAG/'handoff.status.json',{});pids=[sv.get('pid'),sv.get('worker_pid'),handoff.get('pid')]+[r.get('pid') for r in ranks]
processes=[]
for pid in dict.fromkeys(p for p in pids if p):
    p=Path(f'/proc/{pid}/cmdline');processes.append(dict(pid=pid,command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''))
dev=OUT/'development.jsonl';development=[json.loads(s) for s in dev.read_text().splitlines() if s.strip()] if dev.exists() else []
record=dict(time=time.time(),supervisor=sv,ranks=ranks,handoff=handoff,processes=processes,
    latest=read(OUT/'latest.COMMITTED.json'),best=read(OUT/'best.COMMITTED.json'),development=development,
    manifest=read(OUT/'manifest.json',{}),free_bytes=shutil.disk_usage(ROOT).free,
    gpus=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True))
(DIAG/'last_observation.json').write_text(json.dumps(record,indent=2))
print(json.dumps({**{k:v for k,v in record.items() if k not in ('ranks','manifest','development','handoff')},
    'handoff':{k:handoff.get(k) for k in ('state','pid','error','supervisor_pid','source')},
    'ranks':[{k:r.get(k) for k in ('pid','state','update','loss','branch_observation_accuracy','hybrid_branch_observation_accuracy','gradient_norm','time')} for r in ranks],
    'development':[dict(update=r['update'],successes=r['successes'],episodes=r['episodes']) for r in development]}))
