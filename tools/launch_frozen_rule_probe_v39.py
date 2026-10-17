import json,os,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'frozen_rule_learning_v39';out.mkdir(exist_ok=True)
script=base/'symbolic_paired_v12/scripts/probe_frozen_rule_gates_v39.py'
assert not (out/'launcher.json').exists() and not (out/'result.json').exists()
free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>12000
with (out/'driver.log').open('ab') as log:
    child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(script)],cwd=script.parents[1],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=log,stderr=log,start_new_session=True)
record={'pid':child.pid,'time':time.time(),'gpu':7,'script':str(script)}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
