import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'critic_isolation_rl_v50';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect original handles instead of duplicate launch'
source=b/'recovery_rule_learning_v49/without_previous_action'
assert json.loads((source/'status.json').read_text())['state']=='complete'
assert json.loads((source/'development/summary.json').read_text())['successes']==28
assert shutil.disk_usage(b).free>3.4*2**30
src=out/'src';src.mkdir(exist_ok=True)
shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(b/'train_critic_isolation_rl_v50.py',src/'train.py')
for gpu in [1,2]:
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>18000
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent;arm=os.environ['CRITIC_ARM'];folder=out/arm
with (folder/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py'),'--arm',arm],cwd=out/'src',stdout=log,stderr=log)
 state={'supervisor_pid':os.getpid(),'pid':p.pid,'attempt':0,'time':time.time(),'state':'running'}
 (folder/'supervisor.json').write_text(json.dumps(state,indent=2));code=p.wait()
 state.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time());(folder/'supervisor.json').write_text(json.dumps(state,indent=2))
''')
jobs=[]
for arm,gpu in [('shared_critic',1),('detached_critic',2)]:
 folder=out/arm;folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CRITIC_ARM=arm,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 with (folder/'supervisor.log').open('ab') as log:
  proc=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
 jobs.append({'arm':arm,'gpu':gpu,'supervisor_pid':proc.pid})
result={'time':time.time(),'jobs':jobs,'automatic_restart':False,'script_sha256':hashlib.sha256((src/'train.py').read_bytes()).hexdigest(),
 'source_program':str(source),'source_successes_development':28,'fixed80_used':False,
 'comparison':'Same corrected rules, seed, sample schedule, PPO and gradient clipping. Only critic shared-feature gradient is detached in treatment.'}
(out/'launcher.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
