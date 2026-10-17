import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'action_history_ablation_v47';out.mkdir(exist_ok=True)
script=base/'probe_action_history_rules_v47.py';jobs=[]
for arm,gpu in [('without_previous_action',1),('action_dropout',2)]:
 folder=out/arm
 assert not (folder/'launcher.json').exists(),'Existing launch must be inspected, never duplicated'
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>16000,(gpu,free)
assert shutil.disk_usage(base).free>2.5*2**30
src=out/'src';src.mkdir(exist_ok=True);shutil.copytree(base/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True);shutil.copy2(script,src/'probe.py')
for arm,gpu in [('without_previous_action',1),('action_dropout',2)]:
 folder=out/arm;folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(base/'induced_boolean_rules_v41_deps'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (folder/'driver.log').open('ab') as log:
  process=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(src/'probe.py'),'--arm',arm],cwd=src,env=env,stdout=log,stderr=log,start_new_session=True)
 row={'pid':process.pid,'time':time.time(),'gpu':gpu,'arm':arm,'script':str(src/'probe.py'),'script_sha256':hashlib.sha256((src/'probe.py').read_bytes()).hexdigest(),'data_split_unchanged':True,'original_fixed80_used':False,'automatic_restart':False};(folder/'launcher.json').write_text(json.dumps(row,indent=2));jobs.append(row)
(out/'launcher.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs,indent=2))
