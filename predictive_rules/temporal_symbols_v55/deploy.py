import sys
from pathlib import Path
sys.path.insert(0,'NeSy-VLA/tools')
from predictive_remote import connect
local=Path('NeSy-VLA/predictive_rules/temporal_symbols_v55');root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
c=connect()
try:
 with c.open_sftp() as f:
  for name in ('model.py','data.py','objectives.py','temporal_objective.py','train.py','supervisor.py','planner.py','program_archive.py','evaluate.py','storage.py','handoff.py','launch_handoff.py'):
   f.put(str(local/name),root+'/src_temporal_symbols_v55/'+name)
 _,o,e=c.exec_command('python3 '+root+'/src_temporal_symbols_v55/launch_handoff.py',timeout=30)
 print(o.read().decode());print(e.read().decode());assert o.channel.recv_exit_status()==0
finally:c.close()
