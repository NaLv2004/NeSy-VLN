import json,os,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=root/'src_state512_v25'
assert not source.exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpu.splitlines()}
assert all(memory[g]<300 for g in [3,4,5,7])
status=json.loads((root/'run_evidence_rule_v23/status.json').read_text());assert status['state']=='training_budget_complete'
stats=os.statvfs(root);free=stats.f_bavail*stats.f_frsize/2**30;assert free>25,free
shutil.copytree(root/'src_evidence_rule_v23',source)
print(json.dumps(dict(source=str(source),gpus=gpu,free_gib=free,previous_status=status)))
