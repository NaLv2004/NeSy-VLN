import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from predictive_remote import connect
source=Path(__file__).parent;remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_normal_near_medium_v67'
client=connect()
try:
    _,o,e=client.exec_command('mkdir -p '+remote);assert o.channel.recv_exit_status()==0,e.read().decode()
    with client.open_sftp() as ftp:
        for path in sorted(source.glob('*.py')):
            if path.name!='deploy.py':ftp.put(str(path),remote+'/'+path.name)
    _,o,e=client.exec_command('python3 '+remote+'/launch_baseline.py',timeout=30)
    result=o.read().decode();error=e.read().decode();assert o.channel.recv_exit_status()==0,error
    record=json.loads(result);(source/'baseline_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
finally:client.close()
