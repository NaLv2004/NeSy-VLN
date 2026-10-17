import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_goal_probability_v68';OUT=ROOT/'run_goal_probability_v68'
if (OUT/'supervisor.json').exists():
    old=json.loads((OUT/'supervisor.json').read_text());p=Path('/proc')/str(old['pid'])/'cmdline'
    assert not (p.exists() and p.read_bytes()),'Existing supervisor is still active'
    assert old['state']=='needs_repair','Completed runs must not restart'
assert not (OUT/'result.json').exists()
with (OUT/'launcher.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'supervisor.py')],cwd=SOURCE,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(time=time.time(),pid=p.pid,source=str(SOURCE),run=str(OUT),gpus=[6,7])
(OUT/'training_launch.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
