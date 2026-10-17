import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=root/'src_proof_collection_v8r1';out=root/'counterfactual_proof_v8r1'
if (out/'launchers.json').exists():raise RuntimeError('Collection already launched; inspect same handles')
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
if any(memory[g]>300 for g in [3,4,5]):raise RuntimeError('A requested card is busy; no process stopped')
records=[]
for shard,gpu in enumerate([3,4,5]):
    cmd=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'collect_proof_transitions.py'),
        '--checkpoint',str(root/'run_rollout_proof_v8r1/checkpoint_002000.pt'),'--out',str(out),
        '--episodes','192','--shards','3','--shard',str(shard),'--seed','1643']
    log=(out/f'driver_{shard}.log').open('ab',buffering=0)
    p=subprocess.Popen(cmd,cwd=source,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    records.append(dict(pid=p.pid,gpu=gpu,shard=shard,command=cmd,source=str(source)))
record=dict(time=time.time(),processes=records,total_training_episodes=192)
(out/'launchers.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
