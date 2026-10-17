import hashlib,json,os,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'semantic_tokens_v67';out.mkdir(exist_ok=True)
assert not (out/'pilot_launcher.json').exists()
assert int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>18000
source=b/'pilot_semantic_tokens_v67.py';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (out/'pilot.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source)],env=env,stdout=log,stderr=log,start_new_session=True)
r={'pid':p.pid,'gpu':0,'time':time.time(),'script_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(out/'pilot_launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
