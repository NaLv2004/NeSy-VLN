"""One startup observation of the newly launched formal trainer."""
import json
from pathlib import Path
import subprocess
import time

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'run_action_grounded_v33'
launch=json.loads((root/'action_grounded_v33_launcher.json').read_text())
pid=launch['pid'];proc=Path('/proc',str(pid),'cmdline')
def read(p):return json.loads(p.read_text()) if p.exists() else None
updates=[]
if (out/'train.jsonl').exists():
    for line in (out/'train.jsonl').read_text().splitlines():
        try:updates.append(json.loads(line))
        except json.JSONDecodeError:pass
record=dict(observed_at=time.time(),pid=pid,process_alive=proc.exists(),
    command=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
    status=read(out/'status.json'),latest_commit=read(out/'latest.COMMITTED.json'),
    evidence_of_arbitrary_training_boundaries=sorted({r['branch_program_start'] for r in updates if 'branch_program_start' in r}),
    branch_rule_inverse_values=[dict(update=r['update'],loss=r['branch_rule_inverse_loss'],accuracy=r['branch_rule_inverse_accuracy'])
        for r in updates[-3:] if 'branch_rule_inverse_loss' in r],
    log_tail=(root/'action_grounded_v33.log').read_text(errors='replace')[-2000:],
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
(root/'action_grounded_v33_startup.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
