"""One eight-GPU PPO run; two-GPU numerical preflight is not a control experiment."""
import argparse,fcntl,hashlib,json,os,shutil,socket,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'symbolic_ppo8_v82';out.mkdir(exist_ok=True)
lock=(out/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'launcher.json').exists(),'Inspect existing handles; never duplicate a launch'
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
source=b/'perceptual_rl_v80/joint/best_development'
assert read(source.parent/'status.json')['state']=='complete'
best=read(source.parent/'best_score.json');score=read(source.parent/'evaluations'/f"step_{best['epoch']:06d}"/'summary.json');assert score['successes']>=39 and score['episodes']==80
commit=read(source/'COMMITTED');assert sha(source/commit['checkpoint_file'])==commit['checkpoint_sha256']
source_record={'checkpoint':str(source),'commit':commit,'development':score,'selected_without_fixed80':True}
if (out/'SOURCE.json').exists():assert read(out/'SOURCE.json')==source_record
else:(out/'SOURCE.json').write_text(json.dumps(source_record,indent=2))
assert shutil.disk_usage(out).free>8*2**30
src=out/'src';src.mkdir(exist_ok=True)
def free(gpu):return int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
def port():
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]
if a.prepare_only:
 assert all(free(i)>18000 for i in [1,2])
 shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
 shutil.copy2(b/'distributed_symbolic_v82.py',src/'nesy_vla/distributed_symbolic.py')
 shutil.copy2(b/'train_symbolic_ppo8_v82.py',src/'train.py')
 names=['test_small_checkpoint.py','test_factorized_automaton.py','test_probabilistic_automaton.py','test_automaton_policy_gradient.py','test_sequence_perception.py']
 for name in names:shutil.copy2(b/'symbolic_paired_v12/tests'/name,src/name)
 test=subprocess.run([policy,'-m','pytest',*names,'-q'],cwd=src,capture_output=True,text=True)
 (out/'test_results.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
 master_port=port();jobs=[]
 for rank,gpu in enumerate([1,2]):
  folder=out/'preflight'/f'rank{rank}';folder.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),RANK=str(rank),WORLD_SIZE='2',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(master_port),
           OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',NCCL_DEBUG='WARN')
  log=(folder/'driver.log').open('ab')
  child=subprocess.Popen([policy,'-u',str(src/'train.py'),'--arm','joint','--verify-only'],cwd=src,env=env,stdout=log,stderr=log)
  jobs.append((rank,child,log))
 failed=[]
 for rank,child,log in jobs:
  code=child.wait();log.close()
  if code:failed.append({'rank':rank,'code':code,'tail':(out/'preflight'/f'rank{rank}'/'driver.log').read_text(errors='replace')[-5000:]})
 assert not failed,json.dumps(failed)
 prepared={'time':time.time(),'tests':test.stdout,'trainer_sha256':sha(src/'train.py'),'module_sha256':sha(src/'nesy_vla/distributed_symbolic.py'),
           'verification':[read(out/'preflight'/f'rank{rank}'/'verification.json') for rank in range(2)]}
 (out/'PREPARED.json').write_text(json.dumps(prepared,indent=2));print(json.dumps(prepared,indent=2));raise SystemExit
prepared=read(out/'PREPARED.json');assert sha(src/'train.py')==prepared['trainer_sha256']
assert sha(src/'nesy_vla/distributed_symbolic.py')==prepared['module_sha256']
assert all(free(i)>18000 for i in range(8)),[(i,free(i)) for i in range(8)]
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent;rank=int(os.environ['RANK']);folder=out/f'rank{rank}'
def save(s):
 p=folder/'supervisor.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(s,indent=2));os.replace(q,p)
with (folder/'driver.log').open('ab') as log:
 child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py'),'--arm','joint'],cwd=out/'src',stdout=log,stderr=log)
 s={'rank':rank,'supervisor_pid':os.getpid(),'pid':child.pid,'state':'running','time':time.time()};save(s)
 code=child.wait();s.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time());save(s)
''')
master_port=port();result={'time':time.time(),'world':8,'jobs':[],'updates':120,'global_rollouts_per_update':128,
                         'source':source_record,'automatic_restart':False,'experiment_count':1,'prepare':prepared}
for rank in range(8):
 folder=out/f'rank{rank}';folder.mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),RANK=str(rank),WORLD_SIZE='8',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(master_port),
          OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',NCCL_DEBUG='WARN')
 with (folder/'supervisor.log').open('ab') as log:
  child=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
 result['jobs'].append({'rank':rank,'gpu':rank,'supervisor_pid':child.pid});(out/'launcher.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('prepare','source')},indent=2))
