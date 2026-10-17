import shlex,sys
from pathlib import Path
from predictive_remote import connect
source=Path(__file__).resolve().parents[1]/'predictive_rules';root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';dst=root+'/src_visual_predictability_trial'
c=connect()
try:
    _,o,e=c.exec_command('test -e '+root+'/visual_predictability_trial/launcher.json')
    if o.channel.recv_exit_status()==0:raise RuntimeError('Trial already launched; inspect same handle')
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(dst+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    with c.open_sftp() as ftp:
        for path in list((source/'predictive_nesy').glob('*.py'))+[source/'trial_visual_predictability.py']:
            ftp.put(str(path),dst+'/'+path.relative_to(source).as_posix())
    command='cd '+shlex.quote(dst)+' && CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -u trial_visual_predictability.py'
    _,o,e=c.exec_command(command,timeout=180)
    print(o.read().decode());print(e.read().decode());sys.exit(o.channel.recv_exit_status())
finally:c.close()
