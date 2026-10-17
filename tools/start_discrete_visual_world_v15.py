import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=root/'src_discrete_visual_world_v15';out=root/'run_discrete_visual_world_v15'
assert not out.exists()
pre=json.loads((root/'preflight_discrete_world_v15_large/preflight.json').read_text());assert pre['passed'] and pre['global_batch']==4096
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
mem={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()};assert all(mem[g]<300 for g in [3,4,5,7])
out.mkdir();cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
    str(source/'train_discrete_visual_world_v15.py'),'--out',str(out),'--updates','800','--eval-every','200','--batch-size','4096']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0);proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=proc.pid,time=time.time(),source=str(source),command=cmd,train_gpus=[3,4,5],scope='Discrete-symbol world dynamics component; full navigation parity remains required')
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
