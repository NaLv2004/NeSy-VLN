import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'program_causality_v60';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect existing audit'
for arm in ['bridge_only','goal_programs']:
 s=json.loads((b/'goal_conditioned_automaton_v55'/arm/'supervisor.json').read_text())
 assert s['state']=='complete' and not Path(f'/proc/{s["pid"]}').exists()
free=int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>18000
src=out/'src';src.mkdir(exist_ok=True);shutil.copytree(b/'goal_conditioned_automaton_v55/src/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(b/'audit_program_causality_v60.py',src/'audit.py')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(src/'audit.py')],cwd=src,env=env,stdout=log,stderr=log,start_new_session=True)
r={'pid':p.pid,'time':time.time(),'gpu':6,'automatic_restart':False,'new_navigation_rollouts':0,'script_sha256':hashlib.sha256((src/'audit.py').read_bytes()).hexdigest()}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
