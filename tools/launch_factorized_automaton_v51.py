import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'factorized_automaton_v51';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect original handles instead of duplicating runs'
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
prototype=b/'symbolic_paired_v12';src=out/'src';src.mkdir(exist_ok=True)
shutil.copytree(prototype/'nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(b/'train_factorized_automaton_v51.py',src/'train.py')
shutil.copy2(prototype/'tests/test_factorized_automaton.py',src/'test_factorized_automaton.py')
test=subprocess.run([policy,'-m','pytest',str(src/'test_factorized_automaton.py'),'-q'],cwd=src,capture_output=True,text=True)
(out/'test_results.txt').write_text(test.stdout+test.stderr)
assert test.returncode==0,test.stdout+test.stderr
assert shutil.disk_usage(b).free>3.5*2**30
for gpu in [3,4]:
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>18000,(gpu,free)
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent
arm=os.environ['AUTOMATON_ARM']
folder=out/arm;folder.mkdir(exist_ok=True)
with (folder/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py'),'--arm',arm],cwd=out/'src',stdout=log,stderr=log)
 state={'supervisor_pid':os.getpid(),'pid':p.pid,'time':time.time(),'state':'running','attempt':0}
 (folder/'supervisor.json').write_text(json.dumps(state,indent=2))
 code=p.wait()
 state.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time())
 (folder/'supervisor.json').write_text(json.dumps(state,indent=2))
''')
jobs=[]
for arm,gpu in [('recurrent',3),('reset_memory',4)]:
 folder=out/arm;folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),AUTOMATON_ARM=arm,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 with (folder/'supervisor.log').open('ab') as log:
  proc=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],env=env,cwd=out,stdout=log,stderr=log,start_new_session=True)
 jobs.append({'arm':arm,'supervisor_pid':proc.pid,'gpu':gpu})
result={'time':time.time(),'jobs':jobs,'automatic_restart':False,'script_sha256':hashlib.sha256((src/'train.py').read_bytes()).hexdigest(),
 'module_sha256':hashlib.sha256((src/'nesy_vla/factorized_automaton.py').read_bytes()).hexdigest(),
 'tests':test.stdout,'original_fixed80_used':False,'train_split_unchanged':True,'revision_interval':10,'max_actions':32}
(out/'launcher.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
