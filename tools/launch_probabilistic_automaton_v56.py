import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'probabilistic_automaton_v56';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect existing handles; never duplicate an experiment'
assert (b/'factorized_automaton_v51/recurrent/step_000020/COMMITTED').exists()
for arm in ['recurrent','reset_memory']:
 s=json.loads((b/'factorized_automaton_v51'/arm/'supervisor.json').read_text())
 assert s['state']=='complete' and not Path(f'/proc/{s["pid"]}').exists()
for gpu in [3,4]:
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 assert free>18000
assert shutil.disk_usage(b).free>3.3*2**30
src=out/'src';src.mkdir(exist_ok=True)
shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(b/'train_probabilistic_automaton_v56.py',src/'train.py')
for name in ['test_factorized_automaton.py','test_probabilistic_automaton.py']:
 shutil.copy2(b/'symbolic_paired_v12/tests'/name,src/name)
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
test=subprocess.run([policy,'-m','pytest','test_factorized_automaton.py','test_probabilistic_automaton.py','-q'],cwd=src,capture_output=True,text=True)
(out/'test_results.txt').write_text(test.stdout+test.stderr)
assert test.returncode==0,test.stdout+test.stderr
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent;arm=os.environ['SOFT_STATE_ARM'];folder=out/arm
with (folder/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py'),'--arm',arm],cwd=out/'src',stdout=log,stderr=log)
 s={'supervisor_pid':os.getpid(),'pid':p.pid,'attempt':0,'state':'running','time':time.time()}
 (folder/'supervisor.json').write_text(json.dumps(s,indent=2));code=p.wait()
 s.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time());(folder/'supervisor.json').write_text(json.dumps(s,indent=2))
''')
jobs=[]
for arm,gpu in [('soft_recurrent',3),('soft_reset',4)]:
 folder=out/arm;folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),SOFT_STATE_ARM=arm,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 with (folder/'supervisor.log').open('ab') as log:
  process=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
 jobs.append({'arm':arm,'gpu':gpu,'supervisor_pid':process.pid})
r={'time':time.time(),'jobs':jobs,'automatic_restart':False,
 'hashes':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ['train.py','nesy_vla/factorized_automaton.py','nesy_vla/probabilistic_automaton.py']},
 'tests':test.stdout,'horizon':32,'revision_interval':10,'fixed80_used':False,
 'source':'Same v51 recurrent20 weights as v55, fresh optimizer. Soft marginalization intentionally changes initial action probabilities.',
 'semantics':'Observation predicates are hard integer IDs. Memory is a probability distribution over finite state IDs; explicit conditional transition tables propagate it. Pairwise equality conjunctions infer every action. All state/predicate definitions and rule conclusions learned.',
 'comparison':'Soft recurrence versus reset memory. Existing v55 goal_programs provides hard-state control, same data/epochs/optimizer/source.'}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
