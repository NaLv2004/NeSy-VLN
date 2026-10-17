import json,os,signal,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/symbolic_joint_v9')
record=json.loads((root/'controller.json').read_text());pid=record['pid']
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(root/'start_when_ready.py') in cmd
os.kill(pid,signal.SIGSTOP)
if (root/'runs/joint/supervisor.json').exists() or json.loads((root/'allocation.json').read_text())['state']!='waiting_for_completed_evaluation':
    os.kill(pid,signal.SIGCONT);raise RuntimeError('Allocation already occurred; do not replace controller')
os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
history=root/'controller_history';history.mkdir(exist_ok=True)
(root/'controller.json').rename(history/f'controller_{pid}.json')
print(json.dumps({'retired_allocator':pid,'reason':'event-driven saved-update handoff; training untouched','time':time.time()}))
