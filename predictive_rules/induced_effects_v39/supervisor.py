"""Owned two-GPU job, resumable checkpoints, conservative bounded restarts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_induced_effects_v39';OUT=ROOT/'run_induced_effects_v39'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def main():
    OUT.mkdir(exist_ok=True);stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    for attempt in range(3):
        if stop[0]:break
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='6,7',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='2',
            NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTHONUNBUFFERED='1')
        command=[PYTHON,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29739',
            str(SOURCE/'train.py'),'--out',str(OUT),'--updates','10000','--batch-size','128','--branch-batch','64','--eval-every','500']
        if (OUT/'latest.COMMITTED.json').exists():
            commit=json.loads((OUT/'latest.COMMITTED.json').read_text())
            command+=['--resume',str(OUT/commit['filename'])]
        else:
            launch=json.loads((OUT/'launcher.json').read_text())
            command+=['--initialize',launch['source_checkpoint']]
        with (OUT/f'attempt_{attempt}.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(OUT/'supervisor.json',dict(state='running',pid=os.getpid(),worker_pid=child.pid,gpus=[6,7],attempt=attempt,started=time.time()))
            while child.poll() is None and not stop[0]:time.sleep(5)
            if stop[0] and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=120)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.wait()
        if code==0 or stop[0]:
            atomic(OUT/'supervisor.json',dict(state='paused' if stop[0] else 'finished',pid=os.getpid(),exit_code=code,time=time.time()));return
        atomic(OUT/'supervisor.json',dict(state='needs_repair',pid=os.getpid(),exit_code=code,attempt=attempt,time=time.time()))
        # Deterministic data/code failures are surfaced immediately; do not
        # repeatedly occupy GPUs with the same broken run.
        return

if __name__=='__main__':main()
