import json,os,shlex
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1];B='/home/user/xjy/nesy_vla_20261010/consequence_refinement_v93'
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.129.2.34',username='user',password=os.environ['STRIPS_SSH_PASSWORD'],timeout=15,look_for_keys=False,allow_agent=False)
try:
 with client.open_sftp() as s:
  s.put(str(ROOT/'scripts/train_consequence_refinement_v93.py'),B+'/train.py')
 code=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
b=Path({B!r})
assert not (b/'process.json').exists(),'Existing handle must be inspected; never duplicate launch'
files=[b/'train.py',b/'src/nesy_vla/consequence_refinement.py',b/'tests/test_consequence_refinement.py']
for p in files:compile(p.read_text(),str(p),'exec')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
log=(b/'driver.log').open('ab')
p=subprocess.Popen(['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(b/'train.py')],cwd=b,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={{'pid':p.pid,'time':time.time(),'command':str(b/'train.py'),'scope':'Bounded single-GPU symbolic state refinement preflight; live eight-GPU PPO unchanged','source_sha256':{{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}}}}
(b/'process.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
 _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(code),timeout=30)
 text=stdout.read().decode();error=stderr.read().decode();assert stdout.channel.recv_exit_status()==0,error
 record=json.loads(text);(ROOT/'output/consequence_refinement_v93_process.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(text)
finally:client.close()
