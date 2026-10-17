import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=r/'src_raw_grounding_v31';out=r/'run_raw_grounding_v31'
lock=(r/'raw_grounding_v31_train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not out.exists()
old_pid=1750864;assert not Path('/proc',str(old_pid)).exists(),'Existing trainer remains active'
old=json.loads((r/'run_stable_observed_v30/status.json').read_text())
assert old['state']=='training_budget_complete'
used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<256,f'GPU7 occupied ({used} MiB); no unrelated process will be stopped'
available=sum(json.loads(p.read_text())['positions'] for p in (r/'raw_successor_features_v31').glob('counterfactual_*/*.json'))
assert available>=512,available
checkpoint=r/'run_planning_observed_v29/checkpoint_002000.pt'
commit=json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']
args=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'train.py'),
      '--source',str(checkpoint),'--out',str(out),'--batch-size','32','--updates','6000','--eval-every','500']
with (r/'raw_grounding_v31.log').open('xb',buffering=0) as log:
    p=subprocess.Popen(args,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
record=dict(pid=p.pid,command=args,gpu=7,created=time.time(),initial_raw_branch_positions=available,source_sha256=commit['sha256'],
    evaluation_split_unchanged=True,prior_checkpoints_preserved=True,unrelated_processes_untouched=True)
(r/'raw_grounding_v31_launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
