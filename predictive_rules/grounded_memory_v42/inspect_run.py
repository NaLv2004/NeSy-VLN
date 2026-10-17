"""Read-only compact run/process snapshot; invoked at the agreed cadence."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
name=sys.argv[1] if len(sys.argv)>1 else 'run_grounded_memory_v42'
assert name in ('run_learned_persistence_v37','run_discrete_closure_v38','run_grounded_memory_v42')
out=ROOT/name
def read(name):
    p=out/name
    return json.loads(p.read_text()) if p.exists() else None
supervisor=read('supervisor.json');status=read('rank0.status.json')
processes=[]
for pid in (supervisor.get('pid'),supervisor.get('worker_pid'),status.get('pid')):
    if not pid:continue
    p=Path(f'/proc/{pid}')
    processes.append(dict(pid=pid,exists=p.exists(),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace') if p.exists() else None))
dev=out/'development.jsonl'
validations=[json.loads(x) for x in dev.read_text().splitlines() if x.strip()] if dev.exists() else []
print(json.dumps(dict(time=time.time(),run=name,supervisor=supervisor,processes=processes,
    latest=read('latest.COMMITTED.json'),best=read('best.COMMITTED.json'),status=status,
    development=[{k:r.get(k) for k in ('update','successes','planning_failures','false_stops','stop_predicts_changed_observation','observation_prediction_accuracy','actual_steps_per_program')} for r in validations],
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)),indent=2))
