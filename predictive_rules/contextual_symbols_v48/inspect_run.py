import json,shutil,time,subprocess
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_contextual_symbols_v48'
def read(name,default=None):
    path=OUT/name
    return json.loads(path.read_text()) if path.exists() else default
sv=read('supervisor.json',{});states=[read(f'rank{r}.status.json',{}) for r in (0,1)]
pids=[sv.get('pid'),sv.get('worker_pid')]+[s.get('pid') for s in states]
processes=[]
for pid in pids:
    if pid:
        path=Path(f'/proc/{pid}/cmdline')
        processes.append(dict(pid=pid,command=path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else ''))
history=OUT/'development.jsonl'
development=[json.loads(line) for line in history.read_text().splitlines() if line.strip()] if history.exists() else []
record=dict(time=time.time(),supervisor=sv,processes=processes,ranks=states,latest=read('latest.COMMITTED.json'),
    best=read('best.COMMITTED.json'),development=development,free_bytes=shutil.disk_usage(ROOT).free)
record['gpus']=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)
(OUT/'last_observation.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(time=record['time'],supervisor=sv,processes=processes,latest=record['latest'],best=record['best'],
    development=[{k:r.get(k) for k in ('update','successes','episodes','observation_prediction_accuracy','collision_action_rate')} for r in development],
    ranks=[{k:r.get(k) for k in ('pid','phase','update','time')} for r in states],free_bytes=record['free_bytes'],gpus=record['gpus'])))
