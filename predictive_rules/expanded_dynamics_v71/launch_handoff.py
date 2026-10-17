import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_expanded_dynamics_v71';OUT=ROOT/'run_expanded_dynamics_v71'
assert not (OUT/'training_launch.json').exists(),'Do not duplicate a launched experiment'
data=ROOT/'diverse_history_v70';manifest=json.loads((data/'manifest.json').read_text())
assert manifest['tasks']==1200 and manifest['fit_episodes']==960 and manifest['holdout_episodes']==240
source=json.loads((data/'source.COMMITTED.json').read_text());assert source['sha256']=='dba453f4772d7a201ba9b595edd5952d24dd4221bb3b0f41c7d253a15c96f3ed'
OUT.mkdir(exist_ok=True)
shutil.copytree(ROOT/'src_goal_probability_v68/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')}
(OUT/'source_files.json').write_text(json.dumps(code,indent=2))
with (OUT/'launcher.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'handoff.py')],cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=p.pid,source=str(SOURCE),run=str(OUT),gpus=[6,7],state='dependency_wait',
    collection_pid=json.loads((data/'launch.json').read_text())['pid'])
(OUT/'training_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
