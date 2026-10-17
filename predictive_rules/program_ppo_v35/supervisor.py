"""Own-process-only restart supervision with complete committed PPO resume."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_program_ppo_v35'
SOURCE=ROOT/'src_program_ppo_v35'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch=json.loads((ROOT/'program_ppo_v35_launcher.json').read_text())
    stopped={'requested':False}
    signal.signal(signal.SIGTERM,lambda *_:stopped.update(requested=True))
    signal.signal(signal.SIGINT,lambda *_:stopped.update(requested=True))
    children=[]
    for attempt in range(20):
        if stopped['requested']:break
        children=[];logs=[]
        for rank,gpu in enumerate([6,7]):
            log=(OUT/f'rank{rank}_attempt{attempt}.log').open('ab',buffering=0);logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),RANK=str(rank),WORLD_SIZE='2',
                MASTER_ADDR='127.0.0.1',MASTER_PORT='29735',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTHONUNBUFFERED='1')
            env.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
            arguments=[PYTHON,'-u',str(SOURCE/'train_ppo.py'),'--source',launch['source_checkpoint'],
                '--out',str(OUT),'--parallel','64','--episodes','64','--eval-every','10','--updates','0']
            child=subprocess.Popen(arguments,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
        atomic(OUT/'supervisor.json',dict(state='running',pid=os.getpid(),attempt=attempt,
            worker_pids=[p.pid for p in children],gpus=[6,7],started=time.time()))
        while all(p.poll() is None for p in children) and not stopped['requested']:time.sleep(5)
        codes=[p.poll() for p in children]
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        deadline=time.time()+45
        while any(p.poll() is None for p in children) and time.time()<deadline:time.sleep(1)
        for child in children:
            # Descendant simulators share this owned worker group.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        for log in logs:log.close()
        atomic(OUT/'supervisor.json',dict(state='paused' if stopped['requested'] else 'restarting',
            pid=os.getpid(),attempt=attempt,exit_codes=codes,time=time.time(),worker_pids=[p.pid for p in children]))
        if stopped['requested']:return
        time.sleep(10)
    atomic(OUT/'supervisor.json',dict(state='needs_repair',pid=os.getpid(),time=time.time(),attempts=20))


if __name__=='__main__':main()
