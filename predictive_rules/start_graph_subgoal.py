import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');src=root/'src_graph_subgoal_v4';out=root/'run_graph_subgoal_v4'
if (out/'launcher.json').exists():raise RuntimeError('Run already launched; inspect its handle')
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in raw.splitlines()};gpus=[g for g in [3,4,5] if memory[g]<=300]
if not gpus or memory[7]>300:raise RuntimeError('Authorized resources busy; no existing process stopped')
out.mkdir(exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
    PYTHONPATH=str(src),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(gpus)),
    str(src/'train_graph_subgoal.py'),'--out',str(out),'--initialize',str(root/'run_graph_subgoal_v3/checkpoint_000400.pt'),
    '--batch-size','640','--transition-batch-size','32','--updates','6000','--eval-every','400']
for name in ['counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5']:cmd+=['--transitions',str(root/name)]
log=(out/'driver.log').open('ab',buffering=0);p=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),command=cmd,train_gpus=gpus,evaluation_gpu=7,source=str(src))
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
