import shlex,sys
from pathlib import Path
from predictive_remote import connect
source=Path(__file__).resolve().parents[1]/'predictive_rules';root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';dst=root+'/src_subgoal_v2'
c=connect()
try:
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(dst+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    with c.open_sftp() as f:
        for p in list((source/'predictive_nesy').glob('*.py'))+list(source.glob('*subgoal*.py')):f.put(str(p),dst+'/'+p.relative_to(source).as_posix())
    cmd='cd '+shlex.quote(dst)+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -m unittest test_subgoal -v'
    _,o,e=c.exec_command(cmd,timeout=120);print(o.read().decode());print(e.read().decode());sys.exit(o.channel.recv_exit_status())
finally:c.close()
