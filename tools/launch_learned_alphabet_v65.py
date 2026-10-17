import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'learned_alphabet_v65';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists()
assert int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>18000
src=out/'src';src.mkdir(exist_ok=True)
shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(b/'probe_learned_alphabet_v65.py',src/'probe.py')
for name in ['test_probabilistic_automaton.py','test_learned_visual_tokens.py']:shutil.copy2(b/'symbolic_paired_v12/tests'/name,src/name)
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
tests=subprocess.run([python,'-m','pytest','test_probabilistic_automaton.py','test_learned_visual_tokens.py','-q'],cwd=src,capture_output=True,text=True)
(out/'test_results.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0,tests.stdout+tests.stderr
env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen([python,'-u',str(src/'probe.py')],cwd=src,env=env,stdout=log,stderr=log,start_new_session=True)
r={'pid':p.pid,'time':time.time(),'gpu':5,'tests':tests.stdout,
 'hashes':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ['probe.py','nesy_vla/learned_visual_tokens.py']}}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
