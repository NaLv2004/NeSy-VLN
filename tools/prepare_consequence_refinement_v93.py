"""Stage an isolated bounded representation experiment beside the live PPO."""
import json,os,shlex
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1]
B='/home/user/xjy/nesy_vla_20261010';REMOTE=B+'/consequence_refinement_v93'
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.129.2.34',username='user',password=os.environ['STRIPS_SSH_PASSWORD'],timeout=15,look_for_keys=False,allow_agent=False)
try:
 code=f'''from pathlib import Path
import shutil
b=Path({REMOTE!r});b.mkdir(exist_ok=True)
assert not (b/'process.json').exists(),'Already launched'
if not (b/'src').exists():shutil.copytree({B+'/trust_ppo_v90/src'!r},b/'src')
(b/'tests').mkdir(exist_ok=True)
'''
 _,stdout,stderr=c.exec_command('python3 -c '+shlex.quote(code),timeout=30)
 assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
 with c.open_sftp() as s:
  s.put(str(ROOT/'nesy_vla/consequence_refinement.py'),REMOTE+'/src/nesy_vla/consequence_refinement.py')
  s.put(str(ROOT/'tests/test_consequence_refinement.py'),REMOTE+'/tests/test_consequence_refinement.py')
 command='cd '+shlex.quote(REMOTE)+' && PYTHONPATH='+shlex.quote(REMOTE+'/src')+' /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -m pytest -q tests/test_consequence_refinement.py'
 _,stdout,stderr=c.exec_command(command,timeout=60)
 output=stdout.read().decode();error=stderr.read().decode();exitcode=stdout.channel.recv_exit_status()
 record={'exit_code':exitcode,'output':output,'stderr':error};(ROOT/'output/consequence_refinement_v93_tests.json').write_text(json.dumps(record),encoding='utf-8')
 print(json.dumps(record));assert exitcode==0
finally:c.close()
