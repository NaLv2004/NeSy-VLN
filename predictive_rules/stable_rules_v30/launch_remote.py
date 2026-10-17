"""Start one isolated stable-label run after verified v29 completion."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
lock=(r/'stable_observed_v30_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not Path('/proc/1432884').exists(),'The previous trainer is still present; inspect before launching'
status=json.loads((r/'run_planning_observed_v29/status.json').read_text())
assert status['state']=='training_budget_complete' and status['update']==3000
assert json.loads((r/'stable_observed_rules_v30_preflight.json').read_text())['passed']
source=r/'src_stable_observed_v30';out=r/'run_stable_observed_v30';assert not out.exists()
checkpoint=r/'run_planning_observed_v29/checkpoint_002000.pt'
commit=json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']
used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<256,f'GPU7 occupied ({used} MiB), leave other processes alone'
args=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'train.py'),
      '--source',str(checkpoint),'--out',str(out),'--batch-size','64','--updates','3000','--eval-every','500']
with (r/'stable_observed_v30.log').open('xb',buffering=0) as log:
    p=subprocess.Popen(args,cwd=source,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,command=args,gpu=7,created=time.time(),source_checkpoint=str(checkpoint),source_sha256=commit['sha256'],
            all_prior_checkpoints_preserved=True,development_split_unchanged=True,goal_complete=False)
(r/'stable_observed_v30_launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
