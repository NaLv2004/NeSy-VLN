"""Retain an immutable snapshot and start the owned training-data collector."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'onpolicy_successors_v40';source=ROOT/'src_probabilistic_rules_v40'
out.mkdir(exist_ok=True)
if (out/'status.json').exists():
    status=json.loads((out/'status.json').read_text());p=Path(f'/proc/{status["pid"]}/cmdline')
    assert not p.exists() or b'collect_onpolicy.py' not in p.read_bytes(), 'Collector already live'
    assert status['state']!='complete','Collection already completed'
target=out/'policy.pt'
if not target.exists():
    run=ROOT/'run_probabilistic_rules_v40'
    commit=json.loads((run/'latest.COMMITTED.json').read_text())
    # Hard link pins this checkpoint even if rolling retention unlinks its old
    # pathname. No mutation of the original training checkpoint occurs.
    os.link(run/commit['filename'],target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==commit['sha256']
    (out/'policy.COMMITTED.json').write_text(json.dumps(dict(commit,filename='policy.pt')))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (out/'collector.log').open('ab',buffering=0) as log:
    process=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',
        str(source/'collect_onpolicy.py'),'--out',str(out),'--checkpoint',str(target),'--tasks','120','--workers','4'],
        cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launcher.json').write_text(json.dumps(dict(pid=process.pid,time=time.time(),gpu=6)))
print(json.dumps(dict(pid=process.pid,output=str(out),time=time.time())))
