import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');src=root/'src_grounded_aggregate_v10';out=root/'run_grounded_aggregate_v10'
if out.exists():raise RuntimeError('Run already exists; inspect rather than restart')
launch=json.loads((root/'counterfactual_proof_v9/launchers.json').read_text())
for process in launch['processes']:
    assert not (Path('/proc')/str(process['pid'])).exists(), 'Collector handle still exists'
    p=root/'counterfactual_proof_v9'/f"shard_{process['shard']}"/'COMPLETE.json'
    assert p.exists() and json.loads(p.read_text())['completed']==96
assert json.loads((root/'diagnostics/aggregate_rules_preflight.json').read_text())['passed']
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
if any(memory[g]>300 for g in [3,4,5,7]):raise RuntimeError('Authorized resource occupied; no process stopped')
checkpoint=root/'run_grounded_rules_v9/checkpoint_001200.pt';commit=json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==commit['sha256']
names=['counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5','counterfactual_proof_v8r1','counterfactual_proof_v9']
out.mkdir();cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
    str(src/'train_grounded_aggregate_v10.py'),'--out',str(out),'--initialize',str(checkpoint),'--batch-size','640','--transition-batch-size','192',
    '--updates','1200','--eval-every','400','--lr','3e-4']
for name in names:cmd.extend(['--transitions',str(root/name)])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(src),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0);process=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,time=time.time(),source=str(src),command=cmd,train_gpus=[3,4,5],evaluation_gpu=7,initial_checkpoint=commit)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
