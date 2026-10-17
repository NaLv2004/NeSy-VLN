"""Upload isolated source and execute one CPU contract; no training launch."""
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from predictive_remote import connect

here=Path(__file__).resolve().parent
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
target=root+'/src_definition_rules_v56'
names=['model.py','base_model.py','definition_rule.py','data.py','objectives.py','temporal_objective.py',
       'evaluate.py','planner.py','storage.py','program_archive.py','train.py','supervisor.py',
       'inspect_run.py','check_contract.py']
client=connect()
try:
    with client.open_sftp() as ftp:
        for name in names:ftp.put(str(here/name),target+'/'+name)
        ftp.get(root+'/diagnostic_definition_rules_v56/result.json',str(here/'pilot_result.json'))
    command="cd "+shlex.quote(target)+" && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -u check_contract.py"
    _,out,err=client.exec_command(command,timeout=180)
    print(out.read().decode());print(err.read().decode());raise SystemExit(out.channel.recv_exit_status())
finally:client.close()
