"""Stage the specific symmetry repair without changing the completed v93 run."""
import json,os,shlex
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1];B='/home/user/xjy/nesy_vla_20261010';REMOTE=B+'/memory_consequence_v97'
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.129.2.34',username='user',password=os.environ['STRIPS_SSH_PASSWORD'],timeout=15,look_for_keys=False,allow_agent=False)
try:
 code=f'''from pathlib import Path
import shutil
b=Path({REMOTE!r});b.mkdir(exist_ok=True)
assert not (b/'process.json').exists(),'Already launched: preserve immutable source'
if not (b/'src').exists():shutil.copytree({B+'/consequence_refinement_v93/src'!r},b/'src')
(b/'tests').mkdir(exist_ok=True)
'''
 _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(code),timeout=30)
 assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
 with client.open_sftp() as s:
  s.put(str(ROOT/'nesy_vla/consequence_refinement.py'),REMOTE+'/src/nesy_vla/consequence_refinement.py')
  s.put(str(ROOT/'tests/test_consequence_refinement.py'),REMOTE+'/tests/test_consequence_refinement.py')
 command='cd '+shlex.quote(REMOTE)+' && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH='+shlex.quote(REMOTE+'/src')+' /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -m pytest -q tests/test_consequence_refinement.py'
 _,stdout,stderr=client.exec_command(command,timeout=60)
 output=stdout.read().decode();error=stderr.read().decode();exitcode=stdout.channel.recv_exit_status()
 result=dict(exit_code=exitcode,output=output,stderr=error);(ROOT/'output/memory_consequence_v97_tests.json').write_text(json.dumps(result),encoding='utf-8');print(json.dumps(result));assert exitcode==0
finally:client.close()
