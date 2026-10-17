import fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=r/'src_rule_planning_v32';out=r/'run_rule_planning_v32'
lock=(r/'rule_planning_v32_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not out.exists()
previous=r/'run_raw_grounding_v31'
validation=[json.loads(line) for line in (previous/'validation.jsonl').read_text().splitlines()]
assert len(validation)>=5
assert not any(x['successes']>=48 and x['executed_steps_per_program']>=5 for x in validation),'Goal may already be met; audit before another experiment'
best=max(validation,key=lambda x:(x['successes'],x['executed_steps_per_program']))
checkpoint=previous/f"checkpoint_{best['update']:06d}.pt"
commit=json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']==best['checkpoint_sha256']
old=json.loads((r/'raw_grounding_v31_launcher.json').read_text());pid=old['pid'];proc=Path('/proc',str(pid),'cmdline')
last_status=json.loads((previous/'status.json').read_text())
stopped=False
if proc.exists():
    command=proc.read_bytes().replace(b'\0',b' ').decode()
    assert str(r/'src_raw_grounding_v31/train.py') in command and str(previous) in command
    os.killpg(pid,signal.SIGTERM);stopped=True
    for _ in range(80):
        if not proc.exists():break
        time.sleep(.25)
    assert not proc.exists(),'Previous trainer has not stopped; no duplicate launch'
record=dict(state='replaced_for_hard_planning_gradients',time=time.time(),pid=pid,stopped_by_this_launcher=stopped,
    last_status=last_status,all_saved_checkpoints_preserved=True,completed_validation=validation,best=best)
(previous/'replacement_record.json').write_text(json.dumps(record,indent=2))
used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<256,('GPU7 still occupied; do not interrupt other processes',used)
args=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'train.py'),
    '--source',str(checkpoint),'--out',str(out),'--batch-size','32','--updates','6000','--eval-every','500']
with (r/'rule_planning_v32.log').open('xb',buffering=0) as log:
    p=subprocess.Popen(args,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
result=dict(pid=p.pid,command=args,gpu=7,created=time.time(),source_checkpoint=str(checkpoint),source_sha256=commit['sha256'],
    source_development=best,prior_saved_checkpoints_preserved=True,extra_diagnostic_evaluations=0)
(r/'rule_planning_v32_launcher.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
