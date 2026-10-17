import sys,os
from pathlib import Path
sys.path.insert(0,'NeSy-VLA/tools')
from predictive_remote import connect
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';local=Path('NeSy-VLA/predictive_rules/temporal_symbols_v55')
c=connect()
try:
 _,o,e=c.exec_command('mkdir -p '+root+'/src_temporal_symbols_v55 '+root+'/diagnostic_temporal_symbols_v55');assert o.channel.recv_exit_status()==0
 with c.open_sftp() as f:
  for p in local.glob('*.py'):f.put(str(p),root+'/src_temporal_symbols_v55/'+p.name)
 _,o,e=c.exec_command('CUDA_VISIBLE_DEVICES= /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python '+root+'/src_temporal_symbols_v55/check_contract.py',timeout=180)
 print(o.read().decode());print(e.read().decode());assert o.channel.recv_exit_status()==0
finally:c.close()
