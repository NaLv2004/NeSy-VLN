import json, os, subprocess, time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_replay_boundary_v20'; out=root/'replayed_boundary_features_v20'
assert not out.exists()
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}
assert all(memory[g]<300 for g in [3,4,5,7]), memory
out.mkdir(); launchers=[]
for shard,gpu in enumerate([3,4,5,7]):
    command=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'recover_boundary_features_v20.py'),
        '--out',str(out),'--shard',str(shard),'--shards','4']
    log=(out/f'driver_{shard}.log').open('ab',buffering=0)
    p=subprocess.Popen(command,cwd=source,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launchers.append(dict(pid=p.pid,gpu=gpu,shard=shard,command=command,time=time.time()));log.close()
result=dict(source=str(source),out=str(out),launchers=launchers,other_processes_preserved=True,scope='Training-only fixed-action replay; no new policy execution')
(out/'launcher.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
