"""Replace only our v33 trainer; never terminate unrelated GPU users."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_program_ppo_v35';out=root/'run_program_ppo_v35'
lock=(root/'program_ppo_v35_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not out.exists(),'Inspect existing launch before retrying'
assert shutil.disk_usage(root).free>4*2**30,'Insufficient checkpoint reserve'
with socket.socket() as port:
    port.bind(('127.0.0.1',29735))
rows=[json.loads(line) for line in (root/'run_action_grounded_v33/validation.jsonl').read_text().splitlines()]
eligible=[row for row in rows if row['executed_steps_per_program']>=5]
best=max(eligible,key=lambda row:row['successes'])
checkpoint=root/'run_action_grounded_v33'/f'checkpoint_{best["update"]:06d}.pt'
commit=json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']==best['checkpoint_sha256']
old=json.loads((root/'action_grounded_v33_launcher.json').read_text());pid=old['pid']
proc=Path('/proc',str(pid),'cmdline');stopped=False
if proc.exists():
    command=proc.read_bytes().replace(b'\0',b' ').decode()
    assert str(root/'src_action_grounded_v33/train.py') in command and str(root/'run_action_grounded_v33') in command
    assert os.getpgid(pid)==pid
    # All saved v33 checkpoints and its latest committed update remain intact.
    os.killpg(pid,signal.SIGTERM);stopped=True
    for _ in range(100):
        if not proc.exists():break
        time.sleep(.2)
    assert not proc.exists(),'Old trainer has not exited'
(root/'run_action_grounded_v33/replacement_record.json').write_text(json.dumps(dict(
    time=time.time(),state='replaced_by_user_authorized_two_gpu_program_ppo',pid=pid,
    stopped_by_launcher=stopped,all_saved_checkpoints_preserved=True,validation=rows),indent=2))
for gpu in [6,7]:
    for _ in range(20):
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used<256:break
        time.sleep(.5)
    assert used<256,('GPU occupied by another process; refuse launch',gpu,used)
record=dict(created=time.time(),gpus=[6,7],source_checkpoint=str(checkpoint),source_sha256=commit['sha256'],
    source_development=best,parallel_per_gpu=16,episodes_per_update=64,
    fixed_development_episodes=80,no_extra_diagnostic_evaluation=True,prior_checkpoints_preserved=True)
(root/'program_ppo_v35_launcher.json').write_text(json.dumps(record,indent=2))
with (root/'program_ppo_v35_supervisor.log').open('xb',buffering=0) as log:
    child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'supervisor.py')],
        cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record['supervisor_pid']=child.pid
(root/'program_ppo_v35_launcher.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
