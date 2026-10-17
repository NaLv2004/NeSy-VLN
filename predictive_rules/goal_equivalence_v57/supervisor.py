"""Owned two-card contextual-symbol run; preserve checkpoints on every exit."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_goal_equivalence_v57';OUT=ROOT/'run_goal_equivalence_v57'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
    OUT.mkdir(exist_ok=True);stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='6,7',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='2',
        NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTHONUNBUFFERED='1')
    command=[PYTHON,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29757',str(SOURCE/'train.py'),
        '--out',str(OUT),'--updates','2000','--batch-size','128','--branch-batch','64','--eval-every','500',
        '--branch-root',str(ROOT/'onpolicy_successors_v54')]
    if (OUT/'latest.COMMITTED.json').exists():
        commit=json.loads((OUT/'latest.COMMITTED.json').read_text());command+=['--resume',str(OUT/commit['filename'])]
    else:command+=['--initialize',str(ROOT/'diagnostic_goal_equivalence_v57/source.pt')]
    with (OUT/'train.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(OUT/'supervisor.json',dict(state='running',pid=os.getpid(),worker_pid=child.pid,gpus=[6,7],started=time.time()))
        while child.poll() is None and not stop[0]:time.sleep(5)
        if stop[0] and child.poll() is None:
            # The explicit drain tool signals ranks first when a checkpoint is
            # required. This emergency supervisor stop preserves older commits.
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        code=child.wait()
    atomic(OUT/'supervisor.json',dict(state='paused' if stop[0] else ('finished' if code==0 else 'needs_repair'),
        pid=os.getpid(),exit_code=code,time=time.time()))
if __name__=='__main__':main()
