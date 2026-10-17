"""Exact collection-exit dependency; train only after completed committed data."""
import json,os,select,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_expanded_dynamics_v71';OUT=ROOT/'run_expanded_dynamics_v71';DATA=ROOT/'diverse_history_v70'
PY='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def read(p):return json.loads(p.read_text())
def status(**row):
    row.update(pid=os.getpid(),time=time.time());p=OUT/'supervisor.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
try:
    pid=read(DATA/'launch.json')['pid']
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:fd=None
    status(state='waiting_for_collection_exit',collection_pid=pid)
    if fd is not None:select.select([fd],[],[]);os.close(fd)
    collected=read(DATA/'status.json');assert collected['state']=='complete' and collected['completed']==1200,collected
    for p in collected['workers']:
        proc=Path(f'/proc/{p}/cmdline');assert not(proc.exists() and proc.read_bytes()),('Collector worker still active',p)
    for gpu in (6,7):
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert used<500,(gpu,used)
    with (OUT/'training.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([PY,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(SOURCE/'train_joint.py')],cwd=SOURCE,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='6,7',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='2'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status(state='training',child=child.pid,collection=collected);code=child.wait()
    if code:raise RuntimeError(dict(training_exit_code=code))
    result=read(OUT/'result.json');assert result['state']=='complete'
    status(state='finished',exit_code=0)
except Exception as error:
    status(state='needs_repair',error=repr(error),traceback=traceback.format_exc());raise
