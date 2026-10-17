import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=root/'src_state512_v25';out=root/'run_state512_v25'
assert not out.exists()
pre=json.loads((root/'preflight_state512_v25/preflight.json').read_text());assert pre['passed'] and pre['states']==512
assert pre['demonstration_equivalence']['relative_gradient_error']<1e-3 and pre['transition_equivalence']['relative_gradient_error']<1e-3
assert pre['selected_row_argmax_identical'] and pre['AMP_and_runtime_grounding_identical'] and pre['between_revision_neural_call_rejected']
assert all(pre['causal_future_mutation_invariance'].values()) and pre['complete_export_shape']==[512,17,512]
assert pre['memory_peak_gib']<20
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
usage={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()};assert all(usage[g]<300 for g in [3,4,5,7])
initial=root/'initialization_state512_v25r1/initial.pt';assert pre['source_sha256']==hashlib.sha256(initial.read_bytes()).hexdigest()
old=json.loads((root/'run_evidence_rule_v23/manifest.json').read_text());out.mkdir()
command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
    str(source/'train_state512_v25.py'),'--out',str(out),'--initialize',str(initial),'--updates','3000','--batch-size','256',
    '--transition-batch-size','128','--eval-every','1000','--lr','0.0003']
for bank in old['transition_banks']:command.extend(['--transitions',bank['path']])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0)
process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,time=time.time(),source=str(source),command=command,train_gpus=[3,4,5],evaluation_gpu=7,
    states=512,scope='Train fully generated anonymous512-state rules and discrete subgoals; exact ten-action symbolic execution without within-block observations',
    development_interval=1000,updates=3000,initial_checkpoint=str(initial),other_processes_preserved=True)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
