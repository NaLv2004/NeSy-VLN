"""Upload and preflight only. Starting training is a separate explicit step."""
import shlex,sys
from pathlib import Path
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules'
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';source=root+'/src_visual_memory_subgoal_v6'
c=connect()
try:
    _,o,e=c.exec_command('test -e '+root+'/run_visual_memory_subgoal_v6/launcher.json')
    if o.channel.recv_exit_status()==0:raise RuntimeError('Active immutable source must not be overwritten')
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(source+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    paths=list((local/'predictive_nesy').glob('*.py'))+[local/name for name in
        ['train_visual_memory_subgoal.py','evaluate_visual_memory_subgoal.py','start_visual_memory_subgoal.py','test_visual_memory_subgoal.py']]
    with c.open_sftp() as ftp:
        for path in paths:ftp.put(str(path),source+'/'+path.relative_to(local).as_posix())
    command='cd '+shlex.quote(source)+' && OMP_NUM_THREADS=2 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python test_visual_memory_subgoal.py'
    _,o,e=c.exec_command(command,timeout=180)
    print(o.read().decode());print(e.read().decode());sys.exit(o.channel.recv_exit_status())
finally:c.close()
