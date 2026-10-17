import json,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_hybrid_replay_v54';DIAG=ROOT/'diagnostic_hybrid_replay_v54';DATA=ROOT/'onpolicy_successors_v54'
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
sv=read(OUT/'supervisor.json',{});ranks=[read(OUT/f'rank{r}.status.json',{}) for r in (0,1)]
collection=read(DATA/'status.json',{});handoff=read(DIAG/'handoff.status.json',{})
pids=[sv.get('pid'),sv.get('worker_pid'),collection.get('pid'),handoff.get('pid')]+[r.get('pid') for r in ranks]
processes=[]
for pid in dict.fromkeys(p for p in pids if p):
    path=Path(f'/proc/{pid}/cmdline');processes.append(dict(pid=pid,command=path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else ''))
dev=OUT/'development.jsonl'
development=[json.loads(line) for line in dev.read_text().splitlines() if line.strip()] if dev.exists() else []
manifest=read(OUT/'manifest.json',{})
record=dict(time=time.time(),supervisor=sv,ranks=ranks,processes=processes,collection=collection,handoff=handoff,
    latest=read(OUT/'latest.COMMITTED.json'),best=read(OUT/'best.COMMITTED.json'),development=development,
    free_bytes=shutil.disk_usage(ROOT).free,manifest=manifest,
    gpus=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True))
(DIAG/'last_observation.json').write_text(json.dumps(record,indent=2))
print(json.dumps({**{k:v for k,v in record.items() if k not in ('ranks','manifest','development')},
    'ranks':[{k:r.get(k) for k in ('pid','state','update','loss','hybrid_replay_loss','hybrid_current_memory_disagreement','gradient_norm','time')} for r in ranks],
    'development':[dict(update=r['update'],successes=r['successes'],episodes=r['episodes']) for r in development]}))
