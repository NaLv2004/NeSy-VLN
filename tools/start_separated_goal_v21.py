import json,os,subprocess,time,hashlib
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=root/'src_separated_goal_v21';out=root/'run_separated_goal_v21'
assert not out.exists()
pre=json.loads((root/'preflight_separated_goal_v21/preflight.json').read_text())
assert pre['passed'] and pre['gradient_routing']['goal_loss_to_world']==0 and pre['gradient_routing']['world_loss_to_goal']==0
assert pre['gradient_routing']['goal_loss_to_goal']>0 and pre['gradient_routing']['world_loss_to_world']>0
assert all(pre['initial_program_bitwise_identical'].values()) and pre['no_runtime_contract_change']
assert pre['memory_peak_gib']<20
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
usage={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()};assert all(usage[g]<300 for g in [3,4,5,7])
initial=root/'run_replayed_boundary_v20/checkpoint_000600.pt'
assert pre['source_sha256']==hashlib.sha256(initial.read_bytes()).hexdigest()
old=json.loads((root/'run_replayed_boundary_v20/manifest.json').read_text());out.mkdir()
command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
    str(source/'train_separated_goal_v21.py'),'--out',str(out),'--initialize',str(initial),'--updates','1000','--batch-size','512',
    '--transition-batch-size','128','--eval-every','500','--lr','0.0003']
for bank in old['transition_banks']:command.extend(['--transitions',bank['path']])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0)
proc=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=proc.pid,time=time.time(),source=str(source),command=command,train_gpus=[3,4,5],evaluation_gpu=7,
    scope='Actual observed transitions train complete rules; a separate goal branch learns discrete subgoals and symbolic proof actions',
    development_interval=500,updates=1000,initial_checkpoint=str(initial),other_processes_preserved=True)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
