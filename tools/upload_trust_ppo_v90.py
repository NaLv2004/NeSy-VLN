import os
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parents[1];base='/home/user/xjy/nesy_vla_20261010'
files={'scripts/train_trust_ppo_v90.py':'train_trust_ppo_v90.py',
 'scripts/verify_trust_ppo_v90.py':'verify_trust_ppo_v90.py',
 'nesy_vla/symbolic_trust_region.py':'symbolic_trust_region_v90.py',
 'nesy_vla/parallel_rollout.py':'parallel_rollout_v90.py',
 'tests/test_symbolic_trust_region.py':'test_symbolic_trust_region_v90.py'}
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.129.2.34',username='user',password=os.environ['STRIPS_SSH_PASSWORD'],timeout=15,banner_timeout=15,auth_timeout=15,look_for_keys=False,allow_agent=False)
try:
 with c.open_sftp() as s:
  for source,target in files.items():s.put(str(root/source),base+'/'+target)
finally:c.close()
print('Uploaded five isolated preparation files; live training untouched.')
