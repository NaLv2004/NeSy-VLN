import json,time,subprocess
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'diverse_history_v70'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
coordinator=read(OUT/'status.json');ranks=[read(OUT/f'status_rank{r}.json') for r in (0,1)]
pids=set([coordinator.get('pid'),*coordinator.get('workers',[]),*[r.get('pid') for r in ranks]])-{None}
processes=[dict(pid=p,command=Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{p}/cmdline').exists() else '') for p in pids]
logs={}
for name in ('collector_rank0.log','collector_rank1.log'):
    path=OUT/name
    if path.exists():
        with path.open('rb') as f:f.seek(max(0,path.stat().st_size-1800));logs[name]=f.read().decode(errors='replace')
record=dict(time=time.time(),coordinator=coordinator,ranks=ranks,processes=processes,logs=logs,
    gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
    storage=subprocess.check_output(['df','-h',str(OUT)],text=True))
(OUT/'last_observation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
