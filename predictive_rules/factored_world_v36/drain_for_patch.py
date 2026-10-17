import json
import os
from pathlib import Path
import signal
import subprocess
import time
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_factored_world_v36'
status=json.loads((OUT/'supervisor.json').read_text());supervisor=status['pid'];master=status['worker_pid']
def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return ''
assert 'src_factored_world_v36/supervisor.py' in command(supervisor)
workers=[int(x) for x in subprocess.check_output(['pgrep','-P',str(master)],text=True).split()]
assert len(workers)==2 and all('src_factored_world_v36/train.py' in command(pid) for pid in workers)
os.kill(supervisor,signal.SIGSTOP)
for pid in workers:os.kill(pid,signal.SIGTERM)
deadline=time.time()+120
while time.time()<deadline and any(command(pid) for pid in workers):time.sleep(1)
if any(command(pid) for pid in workers):
    os.kill(supervisor,signal.SIGCONT);raise RuntimeError('Still draining, do not restart')
os.kill(supervisor,signal.SIGTERM);os.kill(supervisor,signal.SIGCONT)
deadline=time.time()+20
while time.time()<deadline and command(supervisor):time.sleep(1)
if command(supervisor):raise RuntimeError('Supervisor has not exited')
original=OUT/'manifest.json'
backup=OUT/'manifest_before_observation_only_inverse.json'
if not backup.exists():backup.write_bytes(original.read_bytes())
print(json.dumps({'drained':True,'latest':json.loads((OUT/'latest.COMMITTED.json').read_text())}))
