import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'goal_boundary_data_v66'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
coordinator=read(OUT/'status.json');ranks=[read(OUT/f'status_rank{i}.json') for i in (0,1)]
pids={p for p in [coordinator.get('pid'),*coordinator.get('workers',[])] if p}
result=dict(time=time.time(),coordinator=coordinator,ranks=ranks,
    processes=[dict(pid=p,command=Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{p}/cmdline').exists() else '') for p in pids],
    gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
for rank in (0,1):
    p=OUT/f'collector_rank{rank}.log'
    if p.exists():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1500));result[f'log{rank}']=f.read().decode(errors='replace')
(OUT/'last_observation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
