"""Honor the user's instruction to retire the redundant comparison arm only."""
import json,os,signal,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/perceptual_rl_v80')
folder=b/'action_rules_only';s=json.loads((folder/'supervisor.json').read_text())
supervisor=s['supervisor_pid'];trainer=s['pid'];p=Path(f'/proc/{supervisor}/cmdline')
if p.exists():
 cmd=p.read_bytes();assert str(b/'supervise.py').encode() in cmd
 assert os.getpgid(supervisor)==supervisor
 child=Path(f'/proc/{trainer}/cmdline')
 if child.exists():
  assert b'--arm action_rules_only' in child.read_bytes().replace(b'\0',b' ') and os.getpgid(trainer)==supervisor
 os.killpg(supervisor,signal.SIGTERM)
r={'time':time.time(),'user_requested':True,'reason':'Concentrate resources on one primary PPO experiment; cease rule-only comparison.',
   'supervisor_pid':supervisor,'trainer_pid':trainer,'checkpoint_preserved':json.loads((folder/'latest/COMMITTED').read_text()) if (folder/'latest/COMMITTED').exists() else None}
(folder/'USER_STOP.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
main=json.loads((b/'joint/supervisor.json').read_text());proc=Path(f'/proc/{main["pid"]}/cmdline')
print(json.dumps({'main_trainer_pid':main['pid'],'main_live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None}))
