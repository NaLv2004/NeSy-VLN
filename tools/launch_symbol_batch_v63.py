import hashlib,json,os,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'numerical_symbol_batch_v63';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists()
assert int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>18000
script=b/'audit_symbol_batch_v63.py';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script)],env=env,stdout=log,stderr=log,start_new_session=True)
r={'pid':p.pid,'time':time.time(),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'gpu':0}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
