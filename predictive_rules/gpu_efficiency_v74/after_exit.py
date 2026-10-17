import json,os,select,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'gpu_efficiency_v74';SOURCE=ROOT/'src_gpu_efficiency_v74'
PY='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def status(**r):
    r.update(time=time.time(),pid=os.getpid());(OUT/'supervisor.json').write_text(json.dumps(r,indent=2))
try:
    previous=json.loads((ROOT/'run_continued_evidence_v72/training_launch.json').read_text())
    try:fd=os.pidfd_open(previous['pid'])
    except ProcessLookupError:fd=None
    status(state='waiting_for_own_v72_exit',previous_pid=previous['pid'])
    if fd is not None:select.select([fd],[],[]);os.close(fd)
    memory=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    assert memory<500,memory
    with (OUT/'benchmark.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([PY,'-u',str(SOURCE/'benchmark.py')],cwd=SOURCE,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='2'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status(state='benchmarking',child=p.pid);code=p.wait()
    status(state='complete' if code==0 else 'needs_repair',exit_code=code)
except Exception as e:status(state='needs_repair',error=repr(e),traceback=traceback.format_exc());raise
