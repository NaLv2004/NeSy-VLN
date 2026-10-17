import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'neural_conjunction_learning_v37';out.mkdir(exist_ok=True)
script=base/'symbolic_paired_v12/scripts/probe_neural_conjunctions_v37.py'
assert not (out/'launcher.json').exists() and not (out/'result.json').exists(),'Inspect existing launch before repeating'
free=int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>12000,'GPU6 is not available'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
with (out/'driver.log').open('ab') as log:
    process=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script)],cwd=script.parents[1],env=env,stdout=log,stderr=log,start_new_session=True)
record={'pid':process.pid,'time':time.time(),'script':str(script),'gpu':6,'training_data_only':True}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
