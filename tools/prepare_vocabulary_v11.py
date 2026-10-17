import shlex
from pathlib import Path
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules';root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';source=root+'/src_vocabulary_v11r1'
c=connect()
try:
    _,o,e=c.exec_command('test -e '+source)
    if o.channel.recv_exit_status()==0:raise RuntimeError('Source exists; inspect it before further action')
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(source+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    with c.open_sftp() as ftp:
        for path in list((local/'predictive_nesy').glob('*.py'))+[local/'initialize_vocabulary_definitions.py']:
            ftp.put(str(path),source+'/'+path.relative_to(local).as_posix())
    print('Prepared '+source)
finally:c.close()
