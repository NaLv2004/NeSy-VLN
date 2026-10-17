from pathlib import Path
import shutil
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=r/'src_raw_grounding_v31'
assert not out.exists();out.mkdir()
for name in ['predictive_nesy','evaluate_state512_v25.py','adaptive_model.py','joint_model.py','planning_model.py','hard_interpreter.py']:
    source=r/'src_stable_observed_v30'/name
    if source.is_dir():shutil.copytree(source,out/name)
    else:shutil.copy2(source,out/name)
shutil.copy2(r/'src_replay_boundary_v20/recover_boundary_features_v20.py',out/'recover_boundary_features_v20.py')
shutil.copy2(r/'src_replay_boundary_v20/replay_boundary_worker.py',out/'replay_boundary_worker.py')
print(out)
