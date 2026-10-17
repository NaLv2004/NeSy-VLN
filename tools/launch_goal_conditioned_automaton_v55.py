import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'goal_conditioned_automaton_v55';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Existing handles must be inspected rather than duplicated'
assert (b/'factorized_automaton_v51/recurrent/step_000020/COMMITTED').exists()
for arm in ['shared_critic','detached_critic']:
 old=b/'critic_isolation_rl_v50'/arm;s=json.loads((old/'supervisor.json').read_text());assert s['state']=='complete' and not Path(f'/proc/{s["pid"]}').exists()
for gpu in [1,2]:
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>18000
assert shutil.disk_usage(b).free>3.3*2**30
src=out/'src';src.mkdir(exist_ok=True);shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(b/'train_goal_conditioned_automaton_v55.py',src/'train.py')
shutil.copy2(b/'symbolic_paired_v12/tests/test_factorized_automaton.py',src/'test_factorized_automaton.py')
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
test=subprocess.run([policy,'-m','pytest','test_factorized_automaton.py','-q'],cwd=src,capture_output=True,text=True)
(out/'test_results.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent;arm=os.environ['GOAL_ARM'];folder=out/arm
with (folder/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py'),'--arm',arm],cwd=out/'src',stdout=log,stderr=log)
 s={'supervisor_pid':os.getpid(),'pid':p.pid,'attempt':0,'state':'running','time':time.time()}
 (folder/'supervisor.json').write_text(json.dumps(s,indent=2));code=p.wait()
 s.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time());(folder/'supervisor.json').write_text(json.dumps(s,indent=2))
''')
jobs=[]
for arm,gpu in [('bridge_only',1),('goal_programs',2)]:
 folder=out/arm;folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),GOAL_ARM=arm,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 with (folder/'supervisor.log').open('ab') as log:
  p=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
 jobs.append({'arm':arm,'gpu':gpu,'supervisor_pid':p.pid})
r={'time':time.time(),'jobs':jobs,'automatic_restart':False,'script_sha256':hashlib.sha256((src/'train.py').read_bytes()).hexdigest(),
 'module_sha256':hashlib.sha256((src/'nesy_vla/factorized_automaton.py').read_bytes()).hexdigest(),'tests':test.stdout,
 'warm_start':'v51 recurrent20; both modifications reproduce its initial actions exactly, then use fresh optimizers.',
 'comparison':'Both have an unsaturated goal/history program bridge; treatment also learns separate goal-conditioned program banks.',
 'fixed80_used':False,'horizon':32,'revision_interval':10}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
