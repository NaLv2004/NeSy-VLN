import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_ensemble_rules_v61';SOURCE=ROOT/'src_ensemble_rules_v61'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
OUT.mkdir(exist_ok=True);assert not (OUT/'launcher.json').exists(),'Inspect the original process before retrying'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'probe.log').open('ab',buffering=0) as log:
    p=subprocess.Popen([PYTHON,'-u',str(SOURCE/'probe.py')],cwd=SOURCE,env=env,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,time=time.time(),device='CPU',threads=2,updates=400,formal_training_unchanged=True)
(OUT/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
