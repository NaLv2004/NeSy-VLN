import json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_linear_evidence_v65';OUT=ROOT/'run_linear_evidence_v65'
assert json.loads((ROOT/'run_goal_discrimination_v64/supervisor.json').read_text())['state']=='finished'
assert not (OUT/'supervisor.json').exists()
for index in (6,7):
    used=int(subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    assert used<500,(index,used)
shutil.copytree(ROOT/'src_goal_discrimination_v64/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
OUT.mkdir(exist_ok=True)
with (OUT/'launcher.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=p.pid,source=str(SOURCE),run=str(OUT),gpus=[6,7])
(OUT/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
