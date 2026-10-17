import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'onpolicy_successors_v60';SOURCE=ROOT/'src_onpolicy_refresh_v60';OLD=ROOT/'run_goal_equivalence_v57'
sv=json.loads((OLD/'supervisor.json').read_text());assert sv['state']=='finished' and sv['exit_code']==0
for pid in (2170686,2170687,2170692,2170693):
    p=Path(f'/proc/{pid}/cmdline');assert not p.exists() or not p.read_bytes(),'Old training still live'
assert shutil.disk_usage(ROOT).free>1_900_000_000
gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu,dict(gpu_processes=gpu)
OUT.mkdir(exist_ok=True)
commit=json.loads((OLD/'best.COMMITTED.json').read_text());assert commit['update']==1000
assert commit['sha256']=='58abc7f6a46587f2312412b1848c1e84dc64a374ba725b81f51dd8c0bca880bf'
source=OLD/commit['filename'];assert hashlib.sha256(source.read_bytes()).hexdigest()==commit['sha256']
if not (OUT/'source.pt').exists():os.link(source,OUT/'source.pt')
(OUT/'source.COMMITTED.json').write_text(json.dumps(dict(commit,source_path=str(source)),indent=2))
if not (SOURCE/'predictive_nesy').exists():
    shutil.copytree(ROOT/'src_goal_equivalence_v57/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
launch=OUT/'launcher.json'
assert not launch.exists(),'Inspect the original launched process; never duplicate collection'
log=(OUT/'coordinator.log').open('ab',buffering=0)
p=subprocess.Popen(['python3','-u',str(SOURCE/'coordinate_collection.py')],cwd=SOURCE,
    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),source_commit=commit,tasks=120,gpus=[6,7],training_only=True,
    finalfixed80_used=False,dev80_used_for_collection=False,main_training_started=False)
launch.write_text(json.dumps(record,indent=2));print(json.dumps(record))
