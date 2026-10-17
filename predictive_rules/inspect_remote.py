import json
from pathlib import Path
import subprocess
b = Path('/home/user/xjy/nesy_vla_20261010')
paths = [b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json',
         b/'original_ppo_resume_v99/manifest.json',
         Path('/home/user/xjy/strips_wm_navigation_20261009/data/search_cost_supervision_v3/execution_outcomes.json')]
for p in paths:
    data=json.loads(p.read_text())
    if isinstance(data,list): data={'count':len(data),'first':data[:2]}
    if p.name=='manifest.json': data={'source_commit':data['source_commit']}
    print(str(p),json.dumps(data)[:7000])
print('PROCESSES',subprocess.getoutput('nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader'))
print('POLICYENV',Path('/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python').exists())
