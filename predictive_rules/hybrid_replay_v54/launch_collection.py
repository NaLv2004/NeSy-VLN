import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'onpolicy_successors_v54';SOURCE=ROOT/'src_hybrid_replay_v54';OLD=ROOT/'run_goal_aware_symbols_v52'
assert json.loads((OLD/'supervisor.json').read_text())['state']=='finished'
for pid in (2109713,2109714,2109719,2109720):
    p=Path(f'/proc/{pid}/cmdline')
    assert not p.exists() or not p.read_bytes(), 'Old training still live'
assert shutil.disk_usage(ROOT).free>900_000_000
OUT.mkdir(exist_ok=True)
commit=json.loads((OLD/'best.COMMITTED.json').read_text());assert commit['update']==2000
source=OLD/commit['filename'];assert hashlib.sha256(source.read_bytes()).hexdigest()==commit['sha256']
if not (OUT/'source.pt').exists():os.link(source,OUT/'source.pt')
(OUT/'source.COMMITTED.json').write_text(json.dumps(dict(commit,source_path=str(source)),indent=2))
if not (SOURCE/'predictive_nesy').exists():
    shutil.copytree(ROOT/'src_goal_aware_symbols_v52/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
launch=OUT/'launcher.json'
if launch.exists():
    old=json.loads(launch.read_text());p=Path(f'/proc/{old["pid"]}/cmdline')
    assert not p.exists() or not p.read_bytes(),'Inspect live collection instead of duplicate launch'
    assert json.loads((OUT/'status.json').read_text())['state']!='complete','Completed collection must not repeat'
log=(OUT/'coordinator.log').open('ab',buffering=0)
process=subprocess.Popen(['python3','-u',str(SOURCE/'coordinate_collection.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,time=time.time(),source_commit=commit,tasks=120,gpus=[6,7],training_only=True)
launch.write_text(json.dumps(record,indent=2));print(json.dumps(record))
