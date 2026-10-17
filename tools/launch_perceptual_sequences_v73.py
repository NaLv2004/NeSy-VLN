import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'perceptual_sequences_v73';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists()
script=out/'prepare_perceptual_sequences_v73.py';shutil.copy2(b/'symbolic_paired_v12/scripts/prepare_perceptual_sequences_v73.py',script)
jobs=[]
for gpu,arm in [(0,'frozen'),(1,'tail_1e4')]:
 value=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True);assert int(value.strip())<1000
 folder=out/arm;folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script),'--arm',arm]
 with (folder/'driver.log').open('ab') as log:
  child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 jobs.append({'pid':child.pid,'gpu':gpu,'arm':arm,'command':command})
r={'time':time.time(),'jobs':jobs,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
