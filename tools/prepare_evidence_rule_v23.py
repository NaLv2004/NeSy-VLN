import json,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=root/'src_evidence_rule_v23'
assert not source.exists()
states=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all({int(r.split(',')[0]):int(r.split(',')[1]) for r in states.splitlines()}[g]<300 for g in [3,4,5,7])
assert json.loads((root/'run_separated_goal_v21/status.json').read_text())['state']=='training_budget_complete'
shutil.copytree(root/'src_separated_goal_v21',source)
print(json.dumps(dict(source=str(source),gpus=states)))
