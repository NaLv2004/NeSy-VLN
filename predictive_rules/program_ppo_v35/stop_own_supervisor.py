import json
import os
from pathlib import Path
import signal
import time
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'run_program_ppo_v35';state=json.loads((out/'supervisor.json').read_text())
pid=state['pid'];path=Path('/proc',str(pid),'cmdline')
if path.exists():
    assert str(root/'src_program_ppo_v35/supervisor.py') in path.read_bytes().replace(b'\0',b' ').decode()
    os.kill(pid,signal.SIGTERM)
    for _ in range(55):
        if not path.exists():break
        time.sleep(1)
    assert not path.exists(),'Supervisor still exiting'
print(json.dumps(dict(stopped=pid,checkpoint_preserved=True)))
