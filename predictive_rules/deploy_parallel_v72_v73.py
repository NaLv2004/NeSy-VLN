from pathlib import Path
import ast,json,sys
BASE=Path(__file__).parent;sys.path.insert(0,str(BASE.parent/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
client=connect()
try:
    with client.open_sftp() as ftp:
        for name in ('continued_evidence_v72','compact_history_v73'):
            remote=ROOT+'/src_'+name
            try:ftp.mkdir(remote)
            except OSError:pass
            for p in (BASE/name).glob('*.py'):
                ast.parse(p.read_text(encoding='utf-8-sig'));ftp.put(str(p),remote+'/'+p.name)
        ftp.put(str(BASE/'launch_parallel_v72_v73.py'),ROOT+'/launch_parallel_v72_v73.py')
    _,out,err=client.exec_command('python3 '+ROOT+'/launch_parallel_v72_v73.py',timeout=90)
    result=out.read().decode();error=err.read().decode();assert out.channel.recv_exit_status()==0,error
    record=json.loads(result);(BASE/'parallel_v72_v73_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
finally:client.close()
