import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'run_goal_discrimination_v64';DATA=ROOT/'goal_discrimination_v64'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def tail(p,n=2):return [json.loads(s) for s in p.read_text().splitlines()[-n:]] if p.exists() else []
supervisor=read(OUT/'supervisor.json');ranks=[read(OUT/f'status_rank{i}.json') for i in (0,1)]
pids={p for p in [supervisor.get('pid'),supervisor.get('child'),*[r.get('pid') for r in ranks if r]] if p}
result=dict(time=time.time(),supervisor=supervisor,ranks=ranks,latest=read(OUT/'latest.COMMITTED.json'),
    best=read(OUT/'best.COMMITTED.json'),capacity=read(DATA/'training_capacity.json'),
    heldout=tail(OUT/'goal_holdout.jsonl',3),last_metrics=[tail(OUT/f'rank{i}.jsonl',1) for i in (0,1)],
    processes=[dict(pid=p,command=Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{p}/cmdline').exists() else '') for p in pids],
    gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
    final_result=read(OUT/'result.json'))
for name in ('capacity.log','training.log'):
    p=OUT/name
    if p.exists():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2400));result[name]=f.read().decode(errors='replace')
(OUT/'last_observation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
