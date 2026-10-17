import json,os,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'ppo_update_strength_v89';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists() and not (out/'result.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
with (out/'driver.log').open('ab') as log:
 child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(B/'audit_ppo_update_strength_v89.py')],cwd=B,env=env,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=child.pid,time=time.time(),source='step60 checkpoint and recorded training rollouts61',training_modified=False)
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
