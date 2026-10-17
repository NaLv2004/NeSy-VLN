import json,os,signal,subprocess,time
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');source=r/'src_raw_grounding_v31';out=r/'raw_successor_features_v31'
assert not (out/'parallel_launcher.json').exists()
old=json.loads((r/'raw_successor_features_v31_launcher.json').read_text());pid=old['pid']
path=Path('/proc')/str(pid)/'cmdline'
if path.exists():
    command=path.read_bytes().replace(b'\0',b' ').decode()
    assert str(source/'collect_successor_features.py') in command and str(out) in command
    os.killpg(pid,signal.SIGTERM)
    for _ in range(40):
        if not path.exists():break
        time.sleep(.25)
    assert not path.exists(),'Existing collector has not exited'
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
values={int(line.split(',')[0]):int(line.split(',')[1]) for line in memory.splitlines()}
gpus=[g for g in [3,4,5,6] if values[g]<256]
assert gpus,values
launch=[]
for shard,gpu in enumerate(gpus):
    args=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(source/'collect_successor_features.py'),
          '--out',str(out),'--checkpoint',str(r/'run_planning_observed_v29/checkpoint_002000.pt'),
          '--shard',str(shard),'--shards',str(len(gpus))]
    with (out/f'collector_{shard}.log').open('xb',buffering=0) as log:
        p=subprocess.Popen(args,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    launch.append(dict(pid=p.pid,gpu=gpu,command=args))
record=dict(created=time.time(),workers=launch,scope='training_feature_collection',existing_completed_examples_preserved=True)
(out/'parallel_launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
