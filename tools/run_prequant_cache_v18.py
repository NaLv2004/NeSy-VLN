import os,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');target=root/'src_prequant_diagnostic_v18'
assert not target.exists()
shutil.copytree(root/'src_discrete_rollout_world_v16',target,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(root/'diagnostics/cache_prequant_visual_features_v18.py',target/'cache_prequant_visual_features_v18.py')
subprocess.run(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python',str(target/'cache_prequant_visual_features_v18.py')],
    cwd=target,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2'),check=True)
