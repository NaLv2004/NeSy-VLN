import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'goal_discrimination_v64'
read=lambda p:json.loads(p.read_text()) if p.exists() else {}
prior=read(OUT/'last_observation.json')
if prior and time.time()<prior['time']+630:
    print(json.dumps(dict(state='not_due',next_check=prior['time']+630)));raise SystemExit(0)
launch=read(OUT/'launcher.json');status=read(OUT/'status.json');ranks=[read(OUT/f'status_rank{i}.json') for i in (0,1)]
pids=[launch['pid'],*status.get('workers',[])]
result=dict(time=time.time(),launch=launch,status=status,ranks=ranks,
    processes=[dict(pid=p,command=Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if Path(f'/proc/{p}/cmdline').exists() else '') for p in pids],
    gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
for name in ('collector_rank0.log','collector_rank1.log'):
    with (OUT/name).open('rb') as f:
        f.seek(max(0,(OUT/name).stat().st_size-1600));result[name]=f.read().decode(errors='replace')
(OUT/'last_observation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
