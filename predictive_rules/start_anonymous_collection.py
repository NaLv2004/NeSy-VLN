"""Launch only the two explicitly authorized idle collector devices."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_anonymous_collection_v1';out=root/'counterfactual_anonymous_v5';out.mkdir(exist_ok=True)
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
checkpoint=root/'diagnosis_anonymous_v5/frozen.pt'
if not (source/'VERIFIED.json').exists():raise RuntimeError('Source contract checks missing')
if (out/'launchers.json').exists():raise RuntimeError('Already launched; inspect handles rather than duplicate')
for gpu in [4,5]:
    used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>200:raise RuntimeError(f'GPU{gpu} belongs to another live process')
base=[python,'-u','collect_anonymous_outcomes.py','--checkpoint',str(checkpoint),'--out',str(out),'--episodes','128','--shards','2','--seed','1661']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
subprocess.run(base+['--shard','0','--prepare-only'],cwd=source,env=env,check=True)
records=[]
for shard,gpu in enumerate([4,5]):
    logs=out/f'shard_{shard}';logs.mkdir(exist_ok=True)
    log=(logs/'driver.log').open('ab',buffering=0)
    command=base+['--shard',str(shard)]
    p=subprocess.Popen(command,cwd=source,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,gpu=gpu,time=time.time(),command=command)
    (logs/'launcher.json').write_text(json.dumps(record,indent=2));records.append(record)
(out/'launchers.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
