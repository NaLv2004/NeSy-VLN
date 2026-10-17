import shlex,sys
from pathlib import Path
from predictive_remote import connect
source=Path(__file__).resolve().parents[1]/'predictive_rules';root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';dst=root+'/src_graph_subgoal_v4'
c=connect()
try:
    _,o,e=c.exec_command('test -e '+root+'/run_graph_subgoal_v4/launcher.json')
    if o.channel.recv_exit_status()==0:raise RuntimeError('This snapshot has already launched')
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(dst+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    with c.open_sftp() as ftp:
        files=list((source/'predictive_nesy').glob('*.py'))+list(source.glob('*graph_subgoal*.py'))+[source/'diagnose_graph_exact_alignment.py']
        for path in files:ftp.put(str(path),dst+'/'+path.relative_to(source).as_posix())
    _,o,e=c.exec_command('cd '+shlex.quote(dst)+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -m unittest test_graph_subgoal -v',timeout=120)
    print(o.read().decode());print(e.read().decode());sys.exit(o.channel.recv_exit_status())
finally:c.close()
