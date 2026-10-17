import json,os,subprocess,time
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=r/'raw_successor_features_v31';assert not out.exists()
used=int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<256,f'GPU6 occupied ({used} MiB), no other processes will be stopped'
args=['/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python','-u',str(r/'src_raw_grounding_v31/collect_successor_features.py'),
      '--out',str(out),'--checkpoint',str(r/'run_planning_observed_v29/checkpoint_002000.pt')]
with (r/'raw_successor_features_v31.log').open('xb',buffering=0) as log:
    p=subprocess.Popen(args,cwd=r/'src_raw_grounding_v31',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
record=dict(pid=p.pid,command=args,gpu=6,created=time.time(),task='training_data_feature_recovery',evaluation=False)
(r/'raw_successor_features_v31_launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
