import shlex,sys
from pathlib import Path
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules'
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';source=root+'/src_proof_collection_v11'
c=connect()
try:
    _,o,e=c.exec_command('test -e '+source)
    if o.channel.recv_exit_status()==0:raise RuntimeError('Source exists; inspect without replacing')
    _,o,e=c.exec_command('mkdir -p '+shlex.quote(source+'/predictive_nesy'))
    if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    with c.open_sftp() as ftp:
        for path in list((local/'predictive_nesy').glob('*.py'))+[local/n for n in ['collect_proof_transitions.py','proof_counterfactual_worker.py','proof_collector_simulator.py']]:
            ftp.put(str(path),source+'/'+path.relative_to(local).as_posix())
    cmd='cd '+source+' && CUDA_VISIBLE_DEVICES=3 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python collect_proof_transitions.py --checkpoint '+root+'/run_vocabulary_rules_v11/checkpoint_003200.pt --out '+root+'/counterfactual_proof_v11 --episodes 96 --shards 3 --shard 0 --seed 1844 --prepare-only'
    _,o,e=c.exec_command(cmd,timeout=180);print(o.read().decode());print(e.read().decode());sys.exit(o.channel.recv_exit_status())
finally:c.close()
