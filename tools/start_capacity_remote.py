import json,os,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010')
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
children=[]
for gpu,batch in enumerate([8,10,12,14,16,18]):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    with (root/f'runs/capacity_bf16_{batch}.log').open('wb') as log:
        child=subprocess.Popen([python,'-u','scripts/tune_batch.py','--out',f'runs/capacity_bf16_{batch}.json',
            '--batches',str(batch)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children.append({'gpu':gpu,'batch':batch,'pid':child.pid})
(root/'runs/capacity_jobs.json').write_text(json.dumps(children,indent=2))
print(json.dumps(children))
