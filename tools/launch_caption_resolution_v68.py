import json,os,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'semantic_tokens_v68';out.mkdir(exist_ok=True)
assert not (out/'pilot_launcher.json').exists()
script=b/'symbolic_paired_v12/scripts/caption_symbols_v68.py'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script),'--out',str(out/'pilot'),'--pilot','--batch','16']
with (out/'pilot.log').open('ab') as log:
 child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
r={'pid':child.pid,'gpu':0,'time':time.time(),'command':command};(out/'pilot_launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
