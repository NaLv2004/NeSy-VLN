import shlex,sys
from pathlib import Path
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules'
remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';source=remote+'/src_rollout_proof_v8'
c=connect()
try:
    _,o,e=c.exec_command('test -e '+remote+'/run_rollout_proof_v8/launcher.json')
    if o.channel.recv_exit_status()==0:raise RuntimeError('Do not overwrite an already launched source')
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(source+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    paths=list((local/'predictive_nesy').glob('*.py'))+[local/name for name in
        ['train_rollout_proof.py','evaluate_rollout_proof.py','start_rollout_proof.py','preflight_rollout_proof.py',
         'test_visual_memory_subgoal.py','test_rollout_proof_training.py','test_rule_transition_precision.py']]
    with c.open_sftp() as ftp:
        for path in paths:ftp.put(str(path),source+'/'+path.relative_to(local).as_posix())
    _,o,e=c.exec_command('cd '+shlex.quote(source)+' && CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python preflight_rollout_proof.py',timeout=180)
    print(o.read().decode());print(e.read().decode());sys.exit(o.channel.recv_exit_status())
finally:c.close()
