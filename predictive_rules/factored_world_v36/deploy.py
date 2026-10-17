"""Copy isolated small sources; pause ONLY the owned v35 run after commit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
parser=argparse.ArgumentParser();parser.add_argument('--pause-old',action='store_true');args=parser.parse_args()
source=ROOT/'src_factored_world_v36';source.mkdir(exist_ok=True)
if not (source/'predictive_nesy').exists():
    shutil.copytree(ROOT/'src_program_ppo_v35/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
if args.pause_old:
    status=json.loads((ROOT/'run_program_ppo_v35/supervisor.json').read_text())
    supervisor=status['pid'];workers=status['worker_pids']
    def command(pid):
        try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:return ''
    assert 'program_ppo_v35' in command(supervisor),(supervisor,command(supervisor))
    assert all('src_program_ppo_v35' in command(pid) for pid in workers)
    # Pause the supervisor first so it cannot replace workers being drained.
    os.kill(supervisor,signal.SIGSTOP)
    for pid in workers:os.kill(pid,signal.SIGTERM)
    print(json.dumps(dict(state='draining_owned_v35',supervisor=supervisor,workers=workers)),flush=True)
    deadline=time.time()+150
    while time.time()<deadline and any(command(pid) for pid in workers):time.sleep(2)
    remaining=[pid for pid in workers if command(pid)]
    if remaining:
        os.kill(supervisor,signal.SIGCONT)
        raise RuntimeError('Workers still draining; no new GPU job launched: '+str(remaining))
    os.kill(supervisor,signal.SIGTERM);os.kill(supervisor,signal.SIGCONT)
    report=dict(state='paused',time=time.time(),supervisor=supervisor,workers=workers,
        previous_checkpoint=json.loads((ROOT/'run_program_ppo_v35/latest.COMMITTED.json').read_text()))
    (ROOT/'v35_paused_for_v36.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
print(json.dumps(dict(source=str(source),ready=True)),flush=True)
