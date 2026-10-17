"""Launch the bounded component fit after predecessor termination, without kills."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_discrete_rollout_world_v16';out=root/'run_discrete_rollout_world_v16'
assert not out.exists()
prior=json.loads((root/'run_discrete_visual_world_v15/status.json').read_text())
assert prior['state']=='training_budget_complete' and prior['update']==800
pre=json.loads((root/'preflight_discrete_rollout_world_v16/preflight.json').read_text())
assert pre['passed'] and pre['global_sequence_batch']==128 and pre['global_pair_batch']==1024
contract=json.loads((root/'preflight_discrete_rollout_world_v16/categorical_forward_contract.json').read_text());assert contract['passed']
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
usage={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()};assert all(usage[g]<300 for g in [3,4,5])
out.mkdir();command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
    str(source/'train_discrete_rollout_world_v16.py'),'--out',str(out),'--updates','600','--eval-every','200','--sequence-batch','128','--branch-batch','1024']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    PYTHONPATH=str(source),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0)
process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,time=time.time(),source=str(source),command=command,train_gpus=[3,4,5],
    scope='Training-only discrete world component with autonomous10-step loss; no navigation success claim',
    previous_component=prior,other_processes_preserved=True)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
