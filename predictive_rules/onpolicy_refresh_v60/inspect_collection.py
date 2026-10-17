import json,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'onpolicy_successors_v60'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
launcher=read(OUT/'launcher.json');status=read(OUT/'status.json');ranks=[read(OUT/f'status_rank{r}.json') for r in (0,1)]
pids=set(p for p in [launcher.get('pid'),status.get('pid'),*status.get('workers',[]),*[r.get('pid') for r in ranks]] if p)
processes=[]
for pid in sorted(pids):
    p=Path(f'/proc/{pid}/cmdline');processes.append(dict(pid=pid,command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''))
record=dict(time=time.time(),launcher=launcher,status=status,ranks=ranks,processes=processes,
    manifest=read(OUT/'manifest.json'),free_bytes=shutil.disk_usage(ROOT).free,
    gpus=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
    log_tails={p.name:p.read_text(errors='replace').splitlines()[-8:] for p in OUT.glob('*.log')})
(ROOT/'diagnostic_onpolicy_refresh_v60/last_observation.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='manifest'}))
