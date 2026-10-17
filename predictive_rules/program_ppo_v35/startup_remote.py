"""One startup inspection; routine monitors must wait at least600 seconds."""
import json
import subprocess
import time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'run_program_ppo_v35'
def read(path):return json.loads(path.read_text()) if path.exists() else None
result=dict(observed_at=time.time(),launch=read(root/'program_ppo_v35_launcher.json'),
    supervisor=read(out/'supervisor.json'),status=read(out/'status.json'),latest=read(out/'latest.COMMITTED.json'),
    ranks={str(i):dict(status=read(out/f'rank{i}/status.json'),error=read(out/f'rank{i}/error.json')) for i in range(2)},
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
    log_tails={p.name:p.read_text(errors='replace')[-3500:] for p in sorted(out.glob('rank*_attempt*.log'))[-4:]})
lines=(out/'train.jsonl').read_text().splitlines() if (out/'train.jsonl').exists() else []
result['updates']=[json.loads(line) for line in lines[-3:]]
(root/'program_ppo_v35_startup.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
