"""Wait on the exact collector handle, then validate and fit its saved data."""
import json,os,select,shutil,signal,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_counterfactual_closure_v63';DATA=ROOT/'counterfactual_closure_v63'
OUT=ROOT/'run_counterfactual_closure_v63'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'

def write(row):
    row.update(pid=os.getpid(),time=time.time());p=OUT/'supervisor.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(row,indent=2));t.replace(p)

def main():
    launch=json.loads((DATA/'launcher.json').read_text());pid=launch['pid']
    command=Path(f'/proc/{pid}/cmdline')
    if command.exists() and command.read_bytes():
        assert b'src_counterfactual_closure_v63/coordinate_collection.py' in command.read_bytes()
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:fd=None
        if fd is not None:
            write(dict(state='waiting_collection',collection_pid=pid))
            try:select.select([fd],[],[])
            finally:os.close(fd)
    state=json.loads((DATA/'status.json').read_text())
    assert state['state']=='complete' and state['completed']==600,state
    assert state['exit_codes']==[0,0]
    # Do not interrupt unrelated GPU users, even if new work appeared during
    # collection. Preserve data and require a fresh ownership decision.
    assert not subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert shutil.disk_usage(ROOT).free>350_000_000
    write(dict(state='checking_training_contract'))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (OUT/'objective_contract.log').open('ab',buffering=0) as log:
        subprocess.run([PYTHON,'-u',str(SOURCE/'check_objective.py')],cwd=SOURCE,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    env.update(CUDA_VISIBLE_DEVICES='6,7',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTHONUNBUFFERED='1')
    args=[PYTHON,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29763',
          str(SOURCE/'train_closure.py'),'--updates','2000','--batch','32']
    if (OUT/'latest.COMMITTED.json').exists():
        commit=json.loads((OUT/'latest.COMMITTED.json').read_text())
        args+=['--resume',str(OUT/commit['filename'])]
    with (OUT/'train.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(args,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(dict(state='training',worker_pid=p.pid,gpus=[6,7],updates=2000))
        code=p.wait()
    assert code==0,dict(worker_pid=p.pid,exit_code=code)
    write(dict(state='finished',worker_pid=p.pid,exit_code=code))

try:main()
except Exception as e:
    write(dict(state='needs_repair',error=repr(e),traceback=traceback.format_exc()));raise
