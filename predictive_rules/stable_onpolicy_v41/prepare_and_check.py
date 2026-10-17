import os
from pathlib import Path
import shutil
import subprocess

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_stable_onpolicy_v41'
if not (source/'predictive_nesy').exists():
    shutil.copytree(root/'src_probabilistic_rules_v40/predictive_nesy',source/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
for script in ['check_contracts.py','check_learning_contracts.py']:
    result=subprocess.run([python,str(source/script)],cwd=source,env=env,capture_output=True,text=True,timeout=150)
    print(script,result.stdout,result.stderr,flush=True)
    assert result.returncode==0,script
