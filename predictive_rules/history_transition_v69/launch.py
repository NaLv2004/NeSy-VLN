import json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_history_transition_v69';OUT=ROOT/'run_history_transition_v69'
previous=ROOT/'run_goal_probability_v68'
old=json.loads((previous/'supervisor.json').read_text());assert old['state']=='finished'
for pid in [old['pid'],old.get('child'),*[json.loads((previous/f'status_rank{r}.json').read_text())['pid'] for r in (0,1)]]:
    proc=Path(f'/proc/{pid}/cmdline');assert not(proc.exists() and proc.read_bytes()),('Still running',pid)
assert not (OUT/'training_launch.json').exists(),'Inspect existing run; never silently duplicate it'
for gpu in (6,7):
    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert used<500,(gpu,used)
OUT.mkdir(exist_ok=True)
shutil.copytree(ROOT/'src_goal_probability_v68/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
with (OUT/'launcher.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=p.pid,source=str(SOURCE),run=str(OUT),gpus=[6,7])
(OUT/'training_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
