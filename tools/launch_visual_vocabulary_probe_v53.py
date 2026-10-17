import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'visual_vocabulary_probe_v53';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect original process rather than launch twice'
free=int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>18000
src=out/'src';src.mkdir(exist_ok=True);shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True);shutil.copy2(b/'probe_visual_vocabulary_v53.py',src/'probe.py')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONPATH=str(b/'induced_boolean_rules_v41_deps'))
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(src/'probe.py')],cwd=src,env=env,stdout=log,stderr=log,start_new_session=True)
r={'pid':p.pid,'time':time.time(),'gpu':5,'script_sha256':hashlib.sha256((src/'probe.py').read_bytes()).hexdigest(),'automatic_restart':False,'official_heldout_used':False}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
