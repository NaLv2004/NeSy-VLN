import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'counterfactual_closure_v63';SOURCE=ROOT/'src_counterfactual_closure_v63'
OLD=ROOT/'run_onpolicy_refresh_v60'
assert json.loads((OLD/'supervisor.json').read_text())['state']=='finished'
assert json.loads((ROOT/'diagnostic_ensemble_rules_v61/evaluation.status.json').read_text())['state']=='complete'
assert not subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(ROOT).free>400_000_000
OUT.mkdir(exist_ok=True);assert not (OUT/'launcher.json').exists(),'Inspect existing coordinator before retrying'
commit=json.loads((OLD/'best.COMMITTED.json').read_text())
assert commit['update']==1000 and commit['sha256']=='ae4666e023bf5ee31f9c23962c35b4455cb999285d192ea602b792f02c5a9c19'
source=OLD/commit['filename'];assert hashlib.sha256(source.read_bytes()).hexdigest()==commit['sha256']
if not (OUT/'source.pt').exists():os.link(source,OUT/'source.pt')
(OUT/'source.COMMITTED.json').write_text(json.dumps(dict(commit,source_path=str(source)),indent=2))
if not (SOURCE/'predictive_nesy').exists():
    shutil.copytree(ROOT/'src_onpolicy_refresh_v60_training/predictive_nesy',SOURCE/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
subprocess.run(['python3','-m','compileall','-q',str(SOURCE)],check=True)
with (OUT/'coordinator.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'coordinate_collection.py')],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=p.pid,time=time.time(),source_commit=commit,tasks=600,gpus=[6,7],training_only=True,
    dev80_used_for_collection=False,finalfixed80_used=False,main_training_started=False)
(OUT/'launcher.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
