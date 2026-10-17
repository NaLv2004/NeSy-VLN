import json,os,subprocess,time,shutil,hashlib
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
name='full_history_v79';source=ROOT/('src_'+name);out=ROOT/('run_'+name)
assert not out.exists()
for gpu in ('4','5'):
    used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert used<500,(gpu,used)
out.mkdir()
shutil.copytree(ROOT/'src_expanded_dynamics_v71/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
(out/'source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},indent=2))
with (out/'supervisor.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(source/'supervisor.py')],cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),source=str(source),run=str(out),gpus=[4,5],source_development_successes=50)
(out/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
