import json,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_onpolicy_refresh_v60';DIAG=ROOT/'diagnostic_onpolicy_refresh_v60';DATA=ROOT/'onpolicy_successors_v60'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
now=time.time()
prior=read(DIAG/'training_observation.json')
gate=prior.get('time',1792201528.6444135)+630
assert now>=gate,dict(next_regular_check=gate,now=now)
sv=read(OUT/'supervisor.json');ranks=[read(OUT/f'rank{r}.status.json') for r in (0,1)];handoff=read(DIAG/'handoff.status.json')
pids={2184984,2184985,2184986,2186519}
pids.update(p for p in [sv.get('pid'),sv.get('worker_pid'),*[r.get('pid') for r in ranks]] if p)
processes=[]
for pid in sorted(pids):
    p=Path(f'/proc/{pid}/cmdline');processes.append(dict(pid=pid,command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''))
dev=OUT/'development.jsonl';development=[json.loads(s) for s in dev.read_text().splitlines() if s.strip()] if dev.exists() else []
r=dict(time=now,supervisor=sv,ranks=ranks,handoff=handoff,processes=processes,collection=read(DATA/'status.json'),
    collection_audit=read(DIAG/'collection_audit.json'),contract=read(DIAG/'contract.json'),
    latest=read(OUT/'latest.COMMITTED.json'),best=read(OUT/'best.COMMITTED.json'),development=development,
    free_bytes=shutil.disk_usage(ROOT).free,
    gpus=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
(DIAG/'training_observation.json').write_text(json.dumps(r,indent=2))
print(json.dumps({**{k:v for k,v in r.items() if k not in ('ranks','development','handoff','contract')},
    'handoff':{k:handoff.get(k) for k in ('state','pid','error','supervisor_pid')},
    'ranks':[{k:x.get(k) for k in ('pid','state','update','loss','branch_observation_accuracy','onpolicy_branch_observation_accuracy','gradient_norm','seconds','time')} for x in ranks],
    'development':[dict(update=x['update'],successes=x['successes'],episodes=x['episodes']) for x in development]}))
