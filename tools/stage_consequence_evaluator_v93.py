"""Prepare, but do not launch, the one-checkpoint closed-loop evaluator."""
import os
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1];REMOTE='/home/user/xjy/nesy_vla_20261010/consequence_refinement_v93'
text=(ROOT/'scripts/evaluate_perceptual_fsm.py').read_text(encoding='utf-8')
old="sys.path.insert(0,str(B/'perceptual_fsm_v74/src'))"
assert text.count(old)==1
text=text.replace(old,"sys.path.insert(0,str(B/'consequence_refinement_v93/src'))")
compile(text,'evaluate.py','exec')
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.129.2.34',username='user',password=os.environ['STRIPS_SSH_PASSWORD'],timeout=15,look_for_keys=False,allow_agent=False)
try:
 with client.open_sftp() as s:
  with s.file(REMOTE+'/evaluate.py','w') as f:f.write(text)
  s.put(str(ROOT/'tools/launch_consequence_evaluation_v93_remote.py'),REMOTE+'/launch_evaluation.py')
finally:client.close()
print('One-checkpoint evaluator staged; not launched.')
