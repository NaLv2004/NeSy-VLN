import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'higher_order_capacity_v64';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists()
for gpu in [3,4]:
 assert int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>18000
assert shutil.disk_usage(b).free>3.4*2**30
src=out/'src';src.mkdir(exist_ok=True)
shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(b/'train_higher_order_capacity_v64.py',src/'train.py')
tests=['test_probabilistic_automaton.py','test_higher_order_automaton.py']
for name in tests:shutil.copy2(b/'symbolic_paired_v12/tests'/name,src/name)
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
test=subprocess.run([policy,'-m','pytest',*tests,'-q'],cwd=src,capture_output=True,text=True)
(out/'test_results.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent;arm=os.environ['CAPACITY_ARM'];folder=out/arm
with (folder/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py'),'--arm',arm],cwd=out/'src',stdout=log,stderr=log)
 s={'supervisor_pid':os.getpid(),'pid':p.pid,'attempt':0,'state':'running','time':time.time()}
 (folder/'supervisor.json').write_text(json.dumps(s,indent=2));code=p.wait()
 s.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time());(folder/'supervisor.json').write_text(json.dumps(s,indent=2))
''')
jobs=[]
for arm,gpu in [('pairwise',3),('higher_order',4)]:
 folder=out/arm;folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CAPACITY_ARM=arm,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 with (folder/'supervisor.log').open('ab') as log:
  p=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],env=env,cwd=out,stdout=log,stderr=log,start_new_session=True)
 jobs.append({'arm':arm,'gpu':gpu,'supervisor_pid':p.pid})
r={'time':time.time(),'jobs':jobs,'automatic_restart':False,'tests':test.stdout,'fixed80_used':False,
 'hashes':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ['train.py','nesy_vla/higher_order_automaton.py']}}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
