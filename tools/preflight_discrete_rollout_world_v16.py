import json,os,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_discrete_rollout_world_v16';out=root/'preflight_discrete_rollout_world_v16'
assert not out.exists()
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}[7]<300
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='7';env['OMP_NUM_THREADS']='2'
result=subprocess.run(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python',str(source/'train_discrete_rollout_world_v16.py'),
    '--out',str(out),'--preflight'],cwd=source,env=env,check=True)
print(json.dumps(dict(preflight=json.loads((out/'preflight.json').read_text()),contract=json.loads((out/'categorical_forward_contract.json').read_text()))))
