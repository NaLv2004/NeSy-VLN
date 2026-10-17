import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=root/'src_raw_boundary_v19';out=root/'run_raw_boundary_v19'
assert not out.exists()
pre=json.loads((root/'preflight_raw_boundary_v19/preflight.json').read_text())
assert pre['passed'] and pre['demo_batch']==512 and pre['transition_batch']==128
assert pre['canonical_visual_ids_exact'] and pre['observation_outside_boundary_rejected'] and all(pre['initial_program_bitwise_identical'].values())
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}
assert all(memory[g]<300 for g in [3,4,5,7]),'Do not replace an existing process'
out.mkdir();initial=root/'run_definition_world_v12/checkpoint_001600.pt'
manifest=json.loads((root/'run_definition_world_v12/manifest.json').read_text())
command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
    str(source/'train_raw_boundary_v19.py'),'--out',str(out),'--initialize',str(initial),'--updates','600','--batch-size','512',
    '--transition-batch-size','128','--eval-every','200','--lr','0.0003']
for bank in manifest['transition_banks']:command.extend(['--transitions',bank['path']])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0)
process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,time=time.time(),source=str(source),command=command,train_gpus=[3,4,5],evaluation_gpu=7,
    scope='Complete generated-rule/subgoal navigation candidate; boundary-only raw visual evidence; exact symbolic execution',
    development_interval=200,updates=600,initial_checkpoint=str(initial),other_processes_preserved=True)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
