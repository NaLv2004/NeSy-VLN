import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'program_causality_v60'
old=json.loads((out/'launcher.json').read_text());s=json.loads((out/'status.json').read_text())
assert s['state']=='failed' and 'AssertionError' in s['error'] and not Path(f'/proc/{old["pid"]}').exists()
assert not (out/'result.json').exists()
for source,target in [('launcher.json','failed_launcher.json'),('status.json','failed_status.json'),('src/audit.py','src/failed_audit.py')]:
 assert not (out/target).exists();shutil.copy2(out/source,out/target)
shutil.copy2(b/'audit_program_causality_v60.py',out/'src/audit.py')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (out/'driver.log').open('ab') as log:
 p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(out/'src/audit.py')],cwd=out/'src',env=env,stdout=log,stderr=log,start_new_session=True)
r={'pid':p.pid,'time':time.time(),'gpu':6,'automatic_restart':False,'new_navigation_rollouts':0,'previous_pid':old['pid'],
 'repair':'Use original canonical symbol batch size8 after strict probability replay failure at batch80. Keep strict equality checks; no model or action correction.',
 'script_sha256':hashlib.sha256((out/'src/audit.py').read_bytes()).hexdigest()}
(out/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
