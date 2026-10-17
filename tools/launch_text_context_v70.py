import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'text_context_v70';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists()
gpu=3;memory=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True);assert int(memory.strip())<1000
script=out/'encode_caption_text_v70.py';shutil.copy2(b/'symbolic_paired_v12/scripts/encode_caption_text_v70.py',script)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
with (out/'driver.log').open('ab') as log:
 child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script)],env=env,
  stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
r={'pid':child.pid,'gpu':gpu,'time':time.time(),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
