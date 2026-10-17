import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');out=B/'trust_ppo_v90';src=out/'src';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists()
(out/'PREPARATION_PROCESS.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time()}))
shutil.copytree(B/'parallel_ppo_v85/src',src,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'),dirs_exist_ok=True)
for source,target in [('train_trust_ppo_v90.py','train.py'),('symbolic_trust_region_v90.py','nesy_vla/symbolic_trust_region.py'),
 ('parallel_rollout_v90.py','nesy_vla/parallel_rollout.py'),('test_symbolic_trust_region_v90.py','test_symbolic_trust_region.py')]:
 shutil.copy2(B/source,src/target)
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
names=['test_symbolic_trust_region.py','test_parallel_rollout.py','test_automaton_policy_gradient.py','test_probabilistic_automaton.py','test_small_checkpoint.py']
test=subprocess.run([policy,'-m','pytest',*names,'-q'],cwd=src,capture_output=True,text=True)
(out/'test_results.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
integration=subprocess.run([policy,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(B/'verify_trust_ppo_v90.py')],cwd=src,env=env,capture_output=True,text=True)
(out/'integration.log').write_text(integration.stdout+integration.stderr);assert integration.returncode==0,integration.stdout+integration.stderr
record=dict(time=time.time(),tests=test.stdout,integration=json.loads((out/'integration.json').read_text()),
 trainer_sha256=hashlib.sha256((src/'train.py').read_bytes()).hexdigest(),
 source_hashes={name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ['nesy_vla/symbolic_trust_region.py','nesy_vla/parallel_rollout.py']})
(out/'PREPARED.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
