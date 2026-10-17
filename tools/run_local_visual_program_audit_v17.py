import os,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');target=root/'src_local_visual_program_v17'
assert not target.exists()
shutil.copytree(root/'src_discrete_rollout_world_v16',target,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['predictive_nesy/local_visual_program.py','audit_local_visual_program_v17.py']:
    shutil.copy2(root/'diagnostics/v17_source'/name,target/name)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2')
subprocess.run(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python',str(target/'audit_local_visual_program_v17.py')],cwd=target,env=env,check=True)
