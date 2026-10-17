import json
import os
from pathlib import Path
import subprocess
import time
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'run_program_ppo_v35'
state=json.loads((out/'supervisor.json').read_text());assert not Path('/proc',str(state['pid']),'cmdline').exists()
assert not (out/'latest.COMMITTED.json').exists(),'Unexpected completed update; preserve and explicitly resume its manifest'
manifest=json.loads((out/'manifest.json').read_text())
(out/'initial_startup_manifest.json').write_text(json.dumps(manifest,indent=2))
manifest.update(parallel_simulators_per_gpu=64,episodes_per_gpu=64,episodes_per_update=128)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
for rank in range(2):
    error=out/f'rank{rank}/error.json'
    if error.exists():error.rename(error.with_name('initial_startup_error.json'))
for gpu in [6,7]:
    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used<256,('GPU still occupied',gpu,used)
with (root/'program_ppo_v35_supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(root/'src_program_ppo_v35/supervisor.py')],
        cwd=root/'src_program_ppo_v35',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=json.loads((root/'program_ppo_v35_launcher.json').read_text())
launch.update(supervisor_pid=child.pid,parallel_per_gpu=64,episodes_per_update=128,
    precision_fix='FP32 program likelihood in both rollout and replay; TF32 disabled',restarted=time.time())
(root/'program_ppo_v35_launcher.json').write_text(json.dumps(launch,indent=2))
print(json.dumps(launch))
