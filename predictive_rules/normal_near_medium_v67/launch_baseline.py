import json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_normal_near_medium_v67';OUT=ROOT/'run_normal_near_medium_v67'
PY='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
assert json.loads((ROOT/'goal_boundary_data_v66/status.json').read_text())['state']=='complete'
assert not (OUT/'baseline_launch.json').exists(),'Inspect existing process instead of relaunching'
for gpu in (6,7):
    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    assert used<500,(gpu,used)
shutil.copytree(ROOT/'src_goal_discrimination_v64/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
OUT.mkdir(exist_ok=True)
with (OUT/'baseline.log').open('ab',buffering=0) as log:
    p=subprocess.Popen([PY,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(SOURCE/'baseline.py')],
        cwd=SOURCE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=p.pid,source=str(SOURCE),run=str(OUT),gpus=[6,7])
(OUT/'baseline_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
