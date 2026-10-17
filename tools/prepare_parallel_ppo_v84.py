"""Prepare code/tests without modifying or stopping the live eight-GPU run."""
import json,shutil,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'parallel_ppo_v84';src=out/'src'
assert not (out/'launcher.json').exists()
out.mkdir(exist_ok=True)
shutil.copytree(B/'symbolic_ppo8_v82/src',src,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'),dirs_exist_ok=True)
shutil.copy2(B/'parallel_rollout_v84.py',src/'nesy_vla/parallel_rollout.py')
shutil.copy2(B/'train_parallel_ppo_v84.py',src/'train.py')
shutil.copy2(B/'test_parallel_rollout_v84.py',src/'test_parallel_rollout.py')
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
test=subprocess.run([policy,'-m','pytest','test_parallel_rollout.py','test_automaton_policy_gradient.py','test_probabilistic_automaton.py','test_small_checkpoint.py','-q'],cwd=src,capture_output=True,text=True)
(out/'test_results.txt').write_text(test.stdout+test.stderr)
assert test.returncode==0,test.stdout+test.stderr
(out/'PREPARED.json').write_text(json.dumps({'time':time.time(),'test_result':test.stdout,'continuation_only':True},indent=2))
print(test.stdout)
