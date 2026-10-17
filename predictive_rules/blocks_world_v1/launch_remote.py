import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/blocks_world_v1')
source=root/'src';run=root/'run_20261016'
assert not run.exists(),'Existing run must be inspected, not restarted or overwritten'
assert shutil.disk_usage(root).free>6*2**30
gpu=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
assert int(gpu)<1000,('GPU3 is occupied',gpu)
run.mkdir()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
log=(run/'driver.log').open('ab',buffering=0)
command=[python,'-u',str(source/'run.py'),'--out',str(run),'--representation-updates','2000','--updates','3000']
p=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),gpu=3,source=str(source),run=str(run),command=command,
            source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')},
            other_processes_preserved=True)
(run/'launcher.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
