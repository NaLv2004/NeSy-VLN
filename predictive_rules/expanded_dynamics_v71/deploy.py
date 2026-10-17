import ast,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from predictive_remote import connect
source=Path(__file__).parent;remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_expanded_dynamics_v71'
for path in source.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
client=connect()
try:
    _,out,err=client.exec_command('mkdir -p '+remote);assert out.channel.recv_exit_status()==0,err.read().decode()
    with client.open_sftp() as ftp:
        for path in sorted(source.glob('*.py')):
            if path.name!='deploy.py':ftp.put(str(path),remote+'/'+path.name)
    _,out,err=client.exec_command('python3 '+remote+'/launch_handoff.py',timeout=30)
    result=out.read().decode();error=err.read().decode();assert out.channel.recv_exit_status()==0,error
    record=json.loads(result);(source/'training_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
finally:client.close()
