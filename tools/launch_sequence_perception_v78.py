"""Freeze, verify and launch the paired full-sequence symbolic experiment."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'sequence_perception_v78';out.mkdir(exist_ok=True)
handle=(out/'launch.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
read=lambda path:json.loads(path.read_text())
assert not (out/'launcher.json').exists(),'Existing launch must be inspected, never duplicated'
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
arms=[('frozen',1),('joint',2)]
for arm,gpu in arms:
    c=read(b/'perceptual_sequences_v73'/'tail_1e4'/'TOKENS_COMMITTED')
    assert c['rows']==24848 and c['episodes']==2578
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>18000,(gpu,free)
assert shutil.disk_usage(b).free>1.0*2**30
src=out/'src';src.mkdir(exist_ok=True)
if args.prepare_only:
    shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    shutil.copy2(b/'train_sequence_perception_v78.py',src/'train.py')
    names=['test_small_checkpoint.py','test_factorized_automaton.py','test_probabilistic_automaton.py','test_learned_visual_tokens.py','test_sequence_perception.py']
    for name in names:shutil.copy2(b/'symbolic_paired_v12/tests'/name,src/name)
    test=subprocess.run([policy,'-m','pytest',*names,'-q'],cwd=src,capture_output=True,text=True)
    (out/'test_results.txt').write_text(test.stdout+test.stderr)
    assert test.returncode==0,test.stdout+test.stderr
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
    result={'time':time.time(),'tests':test.stdout,'integration':{arm:read(out/arm/'integration_verified.json') for arm,gpu in arms},
      'trainer_sha256':hashlib.sha256((src/'train.py').read_bytes()).hexdigest()}
    (out/'PREPARED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit

prepared=read(out/'PREPARED.json')
assert hashlib.sha256((src/'train.py').read_bytes()).hexdigest()==prepared['trainer_sha256']
supervisor=out/'supervise.py'
supervisor.write_text('''import json,os,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent;arm=os.environ['PERCEPTUAL_ARM'];folder=out/arm
def save(state):
    target=folder/'supervisor.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));os.replace(temp,target)
with (folder/'driver.log').open('ab') as log:
    child=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/train.py'),'--arm',arm],cwd=out/'src',stdout=log,stderr=log)
    state={'supervisor_pid':os.getpid(),'pid':child.pid,'attempt':0,'state':'running','time':time.time()};save(state)
    code=child.wait();state.update(state='complete' if code==0 else 'failed',exit_code=code,time=time.time());save(state)
''')
result={'time':time.time(),'jobs':[],'automatic_restart':False,'preparation':prepared,
 'comparison':'Matched whole-sequence training from immutable v74tail epoch10. Last visual block receives gradients through hard tokens in joint arm; fixed in control. Same source and32action/revision10. No fixed80 use.',
 'horizon':32,'revision_interval':10,'original_fixed80_used':False,'epochs':10}
for arm,gpu in arms:
    folder=out/arm
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PERCEPTUAL_ARM=arm,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    with (folder/'supervisor.log').open('ab') as log:
        child=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],cwd=out,env=env,stdout=log,stderr=log,start_new_session=True)
    result['jobs'].append({'arm':arm,'gpu':gpu,'supervisor_pid':child.pid})
    (out/'launcher.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
