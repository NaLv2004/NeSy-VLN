"""Verify matched symbolic PPO before launching, without repeating navigation baselines."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'perceptual_rl_v80';out.mkdir(exist_ok=True)
lock=(out/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
read=lambda p:json.loads(p.read_text())
assert not (out/'launcher.json').exists(),'Inspect existing handles instead of relaunching'
source=b/'observation_inverse_v76/observed_inverse';terminal=read(source/'supervisor.json')
assert terminal['state']=='complete' and not Path(f'/proc/{terminal["pid"]}').exists()
assert read(source/'step_000040/COMMITTED')['checkpoint_sha256']=='a842c20cb147822b6b5321fb79247771009a5d8a3b599e5d2979e2cbc940723e'
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
arms=[('joint',0),('action_rules_only',3)]
for arm,gpu in arms:
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>16000,(gpu,free)
free=shutil.disk_usage(b).free;assert free>1.05*2**30,free
src=out/'src';src.mkdir(exist_ok=True)
if args.prepare_only:
    shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    shutil.copy2(b/'train_perceptual_rl_v80.py',src/'train.py')
    names=['test_small_checkpoint.py','test_factorized_automaton.py','test_probabilistic_automaton.py','test_automaton_policy_gradient.py','test_observation_inverse.py']
    for name in names:shutil.copy2(b/'symbolic_paired_v12/tests'/name,src/name)
    test=subprocess.run([policy,'-m','pytest',*names,'-q'],cwd=src,capture_output=True,text=True)
    (out/'test_results.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
    jobs=[]
    for arm,gpu in arms:
        folder=out/arm;folder.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
        log=(folder/'integration.log').open('ab')
        child=subprocess.Popen([policy,'-u',str(src/'train.py'),'--arm',arm,'--verify-only'],cwd=src,env=env,stdout=log,stderr=log)
        jobs.append((arm,child,log))
    failures=[]
    for arm,child,log in jobs:
        code=child.wait();log.close()
        if code:failures.append({'arm':arm,'code':code,'tail':(out/arm/'integration.log').read_text(errors='replace')[-5000:]})
    assert not failures,json.dumps(failures)
    r={'time':time.time(),'tests':test.stdout,'integration':{arm:read(out/arm/'verification.json') for arm,gpu in arms},
       'trainer_sha256':hashlib.sha256((src/'train.py').read_bytes()).hexdigest()}
    (out/'PREPARED.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));raise SystemExit
prepared=read(out/'PREPARED.json');assert hashlib.sha256((src/'train.py').read_bytes()).hexdigest()==prepared['trainer_sha256']
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent;arm=os.environ['PERCEPTUAL_RL_ARM'];folder=out/arm
def save(s):
 p=folder/'supervisor.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(s,indent=2));os.replace(q,p)
with (folder/'driver.log').open('ab') as log:
 child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py'),'--arm',arm],cwd=out/'src',stdout=log,stderr=log)
 s={'supervisor_pid':os.getpid(),'pid':child.pid,'attempt':0,'state':'running','time':time.time()};save(s)
 code=child.wait();s.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time());save(s)
''')
r={'time':time.time(),'jobs':[],'automatic_restart':False,'preparation':prepared,'updates':40,
   'comparison':'Same development-selected v76 observed_inverse40 and frozen learned visual alphabet. Joint PPO vs action-rule-only PPO, matched source and sampler.',
   'source_development_success':35,'horizon':32,'revision_interval':10,'original_fixed80_used_for_training':False}
for arm,gpu in arms:
    folder=out/arm;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PERCEPTUAL_RL_ARM=arm,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    with (folder/'supervisor.log').open('ab') as log:
        child=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
    r['jobs'].append({'arm':arm,'gpu':gpu,'supervisor_pid':child.pid});(out/'launcher.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
