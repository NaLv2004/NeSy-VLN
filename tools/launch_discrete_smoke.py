"""Start isolated short startup checks only on GPUs verified currently free."""
import os,json,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/discrete_v4')
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={int(a):int(b) for a,b in (line.split(',') for line in usage.strip().splitlines())}
for gpu in (6,7):
    if used[gpu]>1500:raise RuntimeError(f'GPU {gpu} is occupied ({used[gpu]} MiB)')
started=[]
for name,gpu in [('discrete_gru',6),('discrete_vlm',7)]:
    out=root/'runs'/f'{name}_smoke';out.mkdir(parents=True,exist_ok=True)
    if (out/'process.json').exists():
        pid=json.loads((out/'process.json').read_text())['pid']
        if Path(f'/proc/{pid}/cmdline').exists():raise RuntimeError('Startup check already has a process')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command=[python,'-u','scripts/train.py','--config',f'configs/{name}.json',
        '--out',str(out),'--smoke','--updates','2']
    with (out/'launch.log').open('ab') as log:
        p=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,gpu=gpu,command=command,time=time.time(),run=str(out))
    (out/'process.json').write_text(json.dumps(record,indent=2));started.append(record)
print(json.dumps(started,indent=2))
