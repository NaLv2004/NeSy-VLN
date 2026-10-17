import shlex,sys
from pathlib import Path
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules'
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
source=root+'/src_anonymous_separation_trial'
client=connect()
try:
    _,out,err=client.exec_command('test -e '+source)
    if out.channel.recv_exit_status()==0:raise RuntimeError('Prototype already prepared; inspect instead of overwriting')
    _,out,err=client.exec_command('mkdir -p '+shlex.quote(source+'/predictive_nesy'))
    if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
    with client.open_sftp() as ftp:
        for path in list((local/'predictive_nesy').glob('*.py'))+[local/'trial_anonymous_separation.py',local/'test_anonymous_separation.py']:
            ftp.put(str(path),source+'/'+path.relative_to(local).as_posix())
    command='cd '+shlex.quote(source)+' && CUDA_VISIBLE_DEVICES= /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python test_anonymous_separation.py'
    _,out,err=client.exec_command(command,timeout=60)
    print(out.read().decode());print(err.read().decode());sys.exit(out.channel.recv_exit_status())
finally:client.close()
