import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'spatial_symbol_reader_v59';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists(),'Inspect existing experiment handles'
old=json.loads((b/'temporal_symbol_gradient_v58/launcher.json').read_text())
assert not Path(f'/proc/{old["pid"]}').exists()
assert json.loads((b/'temporal_symbol_gradient_v58/status.json').read_text())['state']=='complete'
free=int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>18000
src=out/'src';src.mkdir(exist_ok=True)
shutil.copytree(b/'symbolic_paired_v12/nesy_vla',src/'nesy_vla',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(b/'probe_spatial_symbol_reader_v59.py',src/'probe.py')
shutil.copy2(b/'symbolic_paired_v12/tests/test_spatial_token_automaton.py',src/'test_spatial_token_automaton.py')
policy='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
test=subprocess.run([policy,'-m','pytest','test_spatial_token_automaton.py','-q'],cwd=src,capture_output=True,text=True)
(out/'test_results.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen([policy,'-u',str(src/'probe.py')],cwd=src,env=env,stdout=log,stderr=log,start_new_session=True)
r={'pid':p.pid,'time':time.time(),'gpu':5,'automatic_restart':False,'tests':test.stdout,
 'hashes':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ['probe.py','nesy_vla/spatial_token_automaton.py']},
 'official_heldout_used':False,'optimization_updates_per_arm':600,'architecture_difference':'Position-specific token embeddings and flat MLP versus shared token alphabet and spatial convolution; same initialized rule tables and generator.'}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
