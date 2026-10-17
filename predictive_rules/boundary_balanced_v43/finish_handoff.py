"""Wait for the already requested graceful exit, then launch exactly once."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
old=ROOT/'run_grounded_memory_v42';source=ROOT/'src_boundary_balanced_v43'
supervisor=json.loads((old/'supervisor.json').read_text())
pids=[supervisor['pid'],supervisor['worker_pid']]+[json.loads((old/f'rank{rank}.status.json').read_text())['pid'] for rank in (0,1)]
def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return b''
deadline=time.time()+600
print(json.dumps(dict(waiting_for_owned_pids=pids,deadline=deadline)),flush=True)
while any(command(pid) for pid in pids):
    if time.time()>=deadline:raise RuntimeError('v42 still live; no restart performed')
    time.sleep(2)
result=dict(time=time.time(),final_commit=json.loads((old/'latest.COMMITTED.json').read_text()),
    best_commit=json.loads((old/'best.COMMITTED.json').read_text()),
    development=[json.loads(line) for line in (old/'development.jsonl').read_text().splitlines() if line],
    verified_all_old_processes_exited=True)
(source/'predecessor_final.json').write_text(json.dumps(result,indent=2))
subprocess.run(['python3',str(source/'retain_predecessor.py')],check=True)
subprocess.run(['python3',str(source/'launch.py')],check=True)
