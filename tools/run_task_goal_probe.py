import os,shutil,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');src=root/'diagnostics/task_goal_probe'
if src.exists():raise RuntimeError('Diagnostic already prepared; inspect prior result')
shutil.copytree(root/'src_rollout_proof_v8r1',src)
shutil.copyfile(root/'diagnostics/graph_interpreter_task_goal.py',src/'predictive_nesy/graph_subgoal_interpreter.py')
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python',str(root/'diagnostics/diagnose_task_subgoal.py'),
    '--checkpoint',str(root/'run_visual_memory_subgoal_v6/checkpoint_000400.pt'),
    '--out',str(root/'diagnostics/task_goal_probe_initial.json')]
subprocess.run(cmd,cwd=src,env=dict(os.environ,PYTHONPATH=str(src),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),check=True)
