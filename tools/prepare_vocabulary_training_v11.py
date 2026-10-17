import shlex
from pathlib import Path
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules';root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';source=root+'/src_vocabulary_rules_v11'
c=connect()
try:
    _,o,e=c.exec_command('test -e '+source)
    if o.channel.recv_exit_status()==0:raise RuntimeError('Source already exists; inspect instead of replacing')
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(source+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    with c.open_sftp() as ftp:
        for path in list((local/'predictive_nesy').glob('*.py'))+[local/name for name in ['train_vocabulary_rules_v11.py','evaluate_rollout_proof.py','preflight_vocabulary_rules.py']]:
            ftp.put(str(path),source+'/'+path.relative_to(local).as_posix())
    print('Prepared '+source)
finally:c.close()
