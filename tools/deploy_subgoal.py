"""Upload the independent subgoal implementation; never modify a live snapshot."""
import sys,shlex
from pathlib import Path
from predictive_remote import connect

source=Path(__file__).resolve().parents[1]/'predictive_rules'
destination='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_subgoal_v1'
client=connect()
try:
    _,out,err=client.exec_command('test -e /home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_subgoal_v1/launcher.json')
    if out.channel.recv_exit_status()==0:raise RuntimeError('Source snapshot already launched; create a new snapshot instead')
    _,out,err=client.exec_command('mkdir -p '+shlex.quote(destination+'/predictive_nesy'))
    if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
    with client.open_sftp() as ftp:
        for p in list((source/'predictive_nesy').glob('*.py'))+list(source.glob('test_subgoal*.py'))+list(source.glob('*subgoal*.py')):
            ftp.put(str(p),destination+'/'+p.relative_to(source).as_posix())
    command='cd '+shlex.quote(destination)+' && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -m unittest test_subgoal -v'
    _,out,err=client.exec_command(command,timeout=120)
    print(out.read().decode());print(err.read().decode());sys.exit(out.channel.recv_exit_status())
finally:client.close()
