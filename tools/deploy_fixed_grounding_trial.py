import shlex, sys
from pathlib import Path
from predictive_remote import connect

local = Path(__file__).resolve().parents[1]/'predictive_rules'
root = '/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
source = root+'/src_fixed_grounding_trial'
client = connect()
try:
    _, out, err = client.exec_command('test -e '+root+'/fixed_grounding_trial/launcher.json')
    if out.channel.recv_exit_status() == 0:
        raise RuntimeError('Trial already launched; do not restart it')
    _, out, err = client.exec_command('mkdir -p '+shlex.quote(source+'/predictive_nesy'))
    if out.channel.recv_exit_status():
        raise RuntimeError(err.read().decode())
    with client.open_sftp() as ftp:
        for path in list((local/'predictive_nesy').glob('*.py'))+[local/'trial_fixed_grounding.py']:
            ftp.put(str(path), source+'/'+path.relative_to(local).as_posix())
    command = 'cd '+shlex.quote(source)+' && CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -u trial_fixed_grounding.py'
    _, out, err = client.exec_command(command, timeout=180)
    print(out.read().decode());print(err.read().decode());sys.exit(out.channel.recv_exit_status())
finally:
    client.close()
