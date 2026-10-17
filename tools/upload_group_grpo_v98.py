import os
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parents[1];remote='/home/user/xjy/nesy_vla_20261010/group_grpo_v98'
files=['scripts/train_group_grpo_v98.py','scripts/verify_group_grpo_v98.py','nesy_vla/group_relative.py',
       'nesy_vla/grouped_rollout.py','tests/test_group_relative.py','tools/prepare_group_grpo_v98_remote.py',
       'tools/launch_group_grpo_v98_remote.py','tools/inspect_group_grpo_v98_remote.py']
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.129.2.34',username='user',password=os.environ['STRIPS_SSH_PASSWORD'],timeout=15,look_for_keys=False,allow_agent=False)
try:
 with c.open_sftp() as s:
  try:s.stat(remote+'/launcher.json')
  except FileNotFoundError:pass
  else:raise RuntimeError('Experiment already launched; do not overwrite it')
  for folder in [remote,remote+'/staging']:
   try:s.stat(folder)
   except FileNotFoundError:s.mkdir(folder)
  for name in files:s.put(str(root/name),remote+'/staging/'+Path(name).name)
finally:c.close()
print('Uploaded isolated GRPO implementation and preflight checks.')
