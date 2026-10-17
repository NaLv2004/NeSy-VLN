import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=root/'src_definition_world_v12';out=root/'run_definition_world_v12'
assert not out.exists(),'Existing run must be inspected, never replaced'
assert json.loads((root/'diagnostics/definition_world_v12_preflight.json').read_text())['passed']
assert json.loads((root/'diagnostics/proof_collection_v11_full_audit.json').read_text())['full_collection_audited']
for p in json.loads((root/'counterfactual_proof_v11/launchers.json').read_text())['processes']:
    assert not (Path('/proc')/str(p['pid'])).exists(),'Collector still running'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in raw.splitlines()}
assert all(memory[g]<300 for g in [3,4,5,7]),'Required resource busy; no process stopped'
initial=root/'initialization_definition_world_v12/initial.pt';commit=json.loads(initial.with_suffix('.COMMITTED.json').read_text())
assert hashlib.sha256(initial.read_bytes()).hexdigest()==commit['sha256']
out.mkdir()
cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
    str(source/'train_definition_world_v12.py'),'--out',str(out),'--initialize',str(initial),'--batch-size','640','--transition-batch-size','192',
    '--updates','2000','--eval-every','400','--lr','3e-4']
for name in ['counterfactual_training_v1','counterfactual_own_policy_v1','counterfactual_anonymous_v5','counterfactual_proof_v8r1','counterfactual_proof_v9','counterfactual_proof_v11']:
    cmd.extend(['--transitions',str(root/name)])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(source),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0);p=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),source=str(source),command=cmd,train_gpus=[3,4,5],evaluation_gpu=7,initial_checkpoint=commit)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
