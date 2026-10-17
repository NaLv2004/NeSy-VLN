import json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_goal_discrimination_v64';OUT=ROOT/'run_goal_discrimination_v64'
prior=json.loads((OUT/'supervisor.json').read_text())
assert prior['state']=='needs_repair' and 'capacity' in prior['error']
p=Path(f"/proc/{prior['pid']}/cmdline");assert not p.exists() or not p.read_bytes()
assert not (OUT/'latest.COMMITTED.json').exists()
assert not subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
shutil.copytree(ROOT/'src_counterfactual_closure_v63/predictive_nesy',SOURCE/'predictive_nesy',dirs_exist_ok=False)
(OUT/'initial_dependency_failure.json').write_text(json.dumps(prior,indent=2))
with (OUT/'supervisor.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'supervise_training.py')],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),repair='Copied unchanged predictive_nesy dependency snapshot; no optimizer updates occurred before repair')
(OUT/'dependency_repair.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
