"""Restart only this run from atomic checkpoints, with visible failure accounting."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nesy_vla.runtime import atomic,read
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--config',default='configs/navigation.json');p.add_argument('--gpus',default='0,1,2,3,4,5')
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
    NCCL_ASYNC_ERROR_HANDLING='1',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
root=Path(__file__).resolve().parents[1];attempt=0
while True:
    if (out/'STOP').exists():break
    if (out/'status.json').exists() and read(out/'status.json').get('state')=='complete':break
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(a.gpus.split(","))}',
        'scripts/train.py','--config',a.config,'--out',str(out)]
    with (out/f'launch_{attempt:03d}.log').open('ab') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(out/'supervisor.json',{'state':'running','attempt':attempt,'pid':process.pid,'time':time.time(),'command':command})
        while process.poll() is None:
            time.sleep(30)
            if (out/'STOP').exists():os.killpg(process.pid,signal.SIGTERM);break
        code=process.wait()
    if code==0:break
    attempt+=1
    atomic(out/'supervisor.json',{'state':'retry_wait' if attempt<=10 else 'needs_attention',
        'attempt':attempt,'exit_code':code,'time':time.time()})
    if attempt>10:raise SystemExit(code)
    time.sleep(min(60,5*attempt))
atomic(out/'supervisor.json',{'state':'stopped' if (out/'STOP').exists() else 'complete','attempt':attempt,'time':time.time()})
