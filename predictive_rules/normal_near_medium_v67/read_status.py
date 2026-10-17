import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'run_normal_near_medium_v67'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
supervisor=read(OUT/'supervisor.json');ranks=[read(OUT/f'status_rank{i}.json') for i in (0,1)]
pids={p for p in [supervisor.get('pid'),supervisor.get('child'),read(OUT/'baseline_launch.json').get('pid'),*[r.get('pid') for r in ranks]] if p}
r=dict(time=time.time(),supervisor=supervisor,ranks=ranks,
    processes=[dict(pid=p,command=Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{p}/cmdline').exists() else '') for p in pids],
    gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
    latest=read(OUT/'latest.COMMITTED.json'),dataset=read(OUT/'dataset.json'),capacity=read(OUT/'capacity.json'))
for phase in ('baseline_v63_dev80','development_best'):
    summary=read(OUT/phase/'summary.json');r[phase]={k:v for k,v in summary.items() if k not in ('curriculum','checkpoint')}
    r[phase+'_ranks']=[read(OUT/phase/f'rank{i}.status.json') for i in (0,1)]
for name in ('training.log','baseline.log','heldout.jsonl'):
    p=OUT/name
    if p.exists():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));r[name]=f.read().decode(errors='replace')
(OUT/'last_observation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
