import shlex,sys
from pathlib import Path
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules';root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';source=root+'/src_grounded_rules_v9'
c=connect()
try:
    _,o,e=c.exec_command('test -e '+source)
    if o.channel.recv_exit_status()==0:raise RuntimeError('Source already prepared; inspect existing state')
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(source+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    with c.open_sftp() as ftp:
        for path in list((local/'predictive_nesy').glob('*.py'))+[local/name for name in ['train_grounded_rules_v9.py','evaluate_rollout_proof.py','preflight_grounded_rules.py']]:
            ftp.put(str(path),source+'/'+path.relative_to(local).as_posix())
    cmd='cd '+shlex.quote(source)+' && CUDA_VISIBLE_DEVICES=7 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python preflight_grounded_rules.py'
    _,o,e=c.exec_command(cmd,timeout=180);print(o.read().decode());print(e.read().decode());sys.exit(o.channel.recv_exit_status())
finally:c.close()
