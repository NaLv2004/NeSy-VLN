import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'temporal_symbol_gradient_v58'
old=json.loads((out/'launcher.json').read_text());status=json.loads((out/'status.json').read_text())
assert status['state']=='failed' and 'cudnn RNN backward' in status['error']
assert not Path(f'/proc/{old["pid"]}').exists() and not (out/'result.json').exists()
for source,target in [('launcher.json','failed_launcher.json'),('status.json','failed_status.json'),('src/audit.py','src/failed_audit.py')]:
 assert not (out/target).exists();shutil.copy2(out/source,out/target)
shutil.copy2(b/'audit_temporal_symbol_gradient_v58.py',out/'src/audit.py')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/audit.py')],cwd=out/'src',env=env,stdout=log,stderr=log,start_new_session=True)
record={'pid':p.pid,'time':time.time(),'gpu':5,'script_sha256':hashlib.sha256((out/'src/audit.py').read_bytes()).hexdigest(),'automatic_restart':False,'previous_pid':old['pid'],'repair':'Enable GRU training mode for diagnostic backward pass; no dropout layers, no optimizer updates, previous process failed before first measurement.'}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
