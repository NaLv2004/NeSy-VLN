import sys,json,ast
from pathlib import Path
BASE=Path(__file__).parent;sys.path.insert(0,str(BASE.parent/'tools'))
from predictive_remote import connect
ROOT='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
c=connect()
try:
    with c.open_sftp() as ftp:
        for name in ('memory_dynamics_v77','joint_memory_dynamics_v78'):
            target=ROOT+'/src_'+name
            ftp.mkdir(target)
            for p in (BASE/name).glob('*.py'):
                ast.parse(p.read_text());ftp.put(str(p),target+'/'+p.name)
        ftp.put(str(BASE/'launch_memory_dynamics_v77_v78.py'),ROOT+'/launch_memory_dynamics_v77_v78.py')
    _,o,e=c.exec_command('python3 '+ROOT+'/launch_memory_dynamics_v77_v78.py',timeout=60)
    result=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err
    (BASE/'memory_dynamics_launch.json').write_text(result);print(result)
finally:c.close()
