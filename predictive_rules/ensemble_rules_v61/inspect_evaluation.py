"""Inspect only this evaluation's exact coordinator and child handles."""
import json, time
from pathlib import Path

OUT = Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diagnostic_ensemble_rules_v61')
def read(p):
    return json.loads(p.read_text()) if p.exists() else {}
status = read(OUT/'evaluation.status.json')
launch = read(OUT/'evaluation.launch.json')
pids = {launch['pid'], *status.get('workers', [])}
processes = []
for pid in sorted(pids):
    p = Path(f'/proc/{pid}/cmdline')
    processes.append(dict(pid=pid, command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''))
logs = {}
for name in ('evaluation.log','export_contract.log','eval_rank0.log','eval_rank1.log'):
    p = OUT/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-3000));logs[name]=f.read().decode(errors='replace')
result = dict(time=time.time(), status=status, launch=launch, processes=processes,
    export_contract=read(OUT/'export_contract.json'), summary=read(OUT/'dev_full80/summary.json'), logs=logs)
(OUT/'evaluation_observation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
