"""Wait for this identified process to exit, without training-status polling."""
import json,os,select,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');run=root/'run_evidence_rule_v23'
launcher=json.loads((run/'launcher.json').read_text());pid=launcher['pid'];proc=Path('/proc')/str(pid)
if proc.exists():
    command=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
    assert 'train_evidence_rule_v23.py' in command and str(run) in command
    fd=os.pidfd_open(pid);started=time.monotonic()
    try:
        while not select.select([fd],[],[],50)[0]:
            print(json.dumps(dict(waiting_for_existing_training_exit=True,pid=pid)),flush=True)
            if time.monotonic()-started>900:raise TimeoutError('Existing job is still live; do not duplicate')
    finally:os.close(fd)
result=json.loads(subprocess.check_output(['python3',str(root/'diagnostics/check_predictive_training.py'),'--run',run.name],text=True))
target=root/'diagnostics/evidence_rule_v23_completed.json';assert not target.exists();target.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],process_alive=result['process_alive'],committed_hash_verified=result['committed_hash_verified'],
    validation=[dict(update=x['update'],successes=x['successes'],episodes=x['episodes']) for x in result['validation']])),flush=True)
