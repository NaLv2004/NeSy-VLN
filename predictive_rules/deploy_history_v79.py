import ast,sys
from pathlib import Path
BASE=Path(__file__).parent;sys.path.insert(0,str(BASE.parent/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
c=connect()
try:
    with c.open_sftp() as ftp:
        target=ROOT+'/src_full_history_v79';ftp.mkdir(target)
        for p in (BASE/'full_history_v79').glob('*.py'):
            ast.parse(p.read_text());ftp.put(str(p),target+'/'+p.name)
        ftp.put(str(BASE/'full_history_launch.py'),ROOT+'/full_history_launch.py')
    _,o,e=c.exec_command('python3 '+ROOT+'/full_history_launch.py',timeout=60)
    result=o.read().decode();error=e.read().decode();assert o.channel.recv_exit_status()==0,error
    (BASE/'history_v79_launch.json').write_text(result);print(result)
finally:c.close()
