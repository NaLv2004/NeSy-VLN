import os,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');target=root/'src_raw_boundary_v19'
assert not target.exists()
shutil.copytree(root/'src_definition_world_v12',target,ignore=shutil.ignore_patterns('__pycache__'))
for source in (root/'stage_raw_boundary_v19').rglob('*.py'):
    destination=target/source.relative_to(root/'stage_raw_boundary_v19');destination.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source,destination)
subprocess.run(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python',str(target/'preflight_raw_boundary_v19.py')],cwd=target,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=True)
