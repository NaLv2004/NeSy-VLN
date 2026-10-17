"""Replace only our failed-to-improve formal run, keeping its saved evidence."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_action_grounded_v33'
out=root/'run_action_grounded_v33'
lock=(root/'action_grounded_v33_launch.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not out.exists(), 'Existing run must be inspected, never duplicate-launched'
previous=root/'run_rule_planning_v32'
results=[json.loads(line) for line in (previous/'validation.jsonl').read_text().splitlines()]
assert len(results)>=4, 'Allow the current formal run to produce existing scheduled evidence first'
assert not any(x['successes']>=48 and x['executed_steps_per_program']>=5 for x in results), 'Audit a potentially achieved goal first'
assert max(x['successes'] for x in results)<24, 'Meaningful improvement warrants continued training before replacement'
candidates=[]
for name in ['run_raw_grounding_v31','run_rule_planning_v32']:
    for line in (root/name/'validation.jsonl').read_text().splitlines():
        result=json.loads(line)
        if result['executed_steps_per_program']>=5:
            candidates.append((name,result))
run,best=max(candidates,key=lambda x:(x[1]['successes'],x[1]['executed_steps_per_program']))
checkpoint=root/run/f"checkpoint_{best['update']:06d}.pt"
committed=json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==committed['sha256']==best['checkpoint_sha256']

old=json.loads((root/'rule_planning_v32_launcher.json').read_text())
pid=old['pid'];proc=Path('/proc',str(pid),'cmdline')
last=json.loads((previous/'status.json').read_text())
stopped=False
if proc.exists():
    command=proc.read_bytes().replace(b'\0',b' ').decode()
    assert str(root/'src_rule_planning_v32/train.py') in command and str(previous) in command
    assert os.getpgid(pid)==pid
    os.killpg(pid,signal.SIGTERM);stopped=True
    for _ in range(80):
        if not proc.exists():break
        time.sleep(.25)
    assert not proc.exists(), 'Trainer still exists; refuse overlapping launch'
(previous/'replacement_record.json').write_text(json.dumps(dict(
    state='replaced_for_rule_inverse_and_causal_branch_training',time=time.time(),pid=pid,
    stopped_by_this_launcher=stopped,last_status=last,completed_validation=results,
    all_saved_checkpoints_preserved=True),indent=2))
used=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<256, ('GPU7 occupied; do not interrupt any other process',used)

arguments=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'train.py'),
    '--source',str(checkpoint),'--out',str(out),'--batch-size','32','--updates','6000','--eval-every','500']
with (root/'action_grounded_v33.log').open('xb',buffering=0) as log:
    trainer=subprocess.Popen(arguments,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
record=dict(pid=trainer.pid,command=arguments,gpu=7,created=time.time(),
    source_checkpoint=str(checkpoint),source_sha256=committed['sha256'],source_development=best,
    source_run=run,prior_saved_checkpoints_preserved=True,extra_diagnostic_evaluations=0,
    objective_changes=['Inverse-action likelihood computed from generated transition rules',
        'All-action branch samples use arbitrary causal program boundaries'])
(root/'action_grounded_v33_launcher.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
