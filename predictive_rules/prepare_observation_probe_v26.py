import json
from pathlib import Path
import shutil
import subprocess

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_state512_v25'
target=root/'src_observation_probe_v26'
gpu=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
if int(gpu.strip())>1000:raise RuntimeError('GPU7 is occupied; preserve other processes')
assert not (target/'predictive_nesy').exists(),'Do not replace an existing source snapshot'
shutil.copytree(source/'predictive_nesy',target/'predictive_nesy')
shutil.copy2(source/'evaluate_state512_v25.py',target/'evaluate_state512_v25.py')
print(json.dumps(dict(source=str(target),gpu=7,other_jobs_preserved=True)))
