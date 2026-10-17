import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'semantic_tokens_v68/corpus';out.mkdir(parents=True,exist_ok=True)
assert not (out/'launcher.json').exists()
assert json.loads((b/'semantic_tokens_v68/pilot/rank0.status.json').read_text())['state']=='complete'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<1000 for line in raw.strip().splitlines()),raw
source=b/'symbolic_paired_v12/scripts/caption_symbols_v68.py';script=out/'caption_symbols_v68.py';shutil.copy2(source,script)
jobs=[]
for gpu in range(8):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
 command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script),'--out',str(out),
  '--rank',str(gpu),'--world','8','--batch','64']
 with (out/f'rank{gpu}.log').open('ab') as log:
  child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 jobs.append({'pid':child.pid,'gpu':gpu,'rank':gpu,'command':command})
result={'time':time.time(),'jobs':jobs,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
(out/'launcher.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
