import json,os,select,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_normal_near_medium_v67';OUT=ROOT/'run_normal_near_medium_v67'
PY='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def status(**row):
    row.update(pid=os.getpid(),time=time.time());p=OUT/'supervisor.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
try:
    baseline=json.loads((OUT/'baseline_launch.json').read_text());pid=baseline['pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists() and proc.read_bytes():
        assert b'baseline.py' in proc.read_bytes()
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:fd=None
        if fd is not None:
            status(state='waiting_for_new80_baseline',baseline_pid=pid)
            select.select([fd],[],[]);os.close(fd)
    assert (OUT/'baseline_v63_dev80/summary.json').exists(),'Baseline did not complete successfully'
    for gpu in (6,7):
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        assert used<500,('Unexpected active GPU process after baseline',gpu,used)
    command=[PY,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(SOURCE/'train_joint.py')]
    if (OUT/'latest.COMMITTED.json').exists():command+=['--resume',json.loads((OUT/'latest.COMMITTED.json').read_text())['filename']]
    with (OUT/'training.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(command,cwd=SOURCE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6,7',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='2'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status(state='training',child=p.pid);code=p.wait()
    if code:raise RuntimeError(dict(stage='training',exit_code=code))
    result=json.loads((OUT/'result.json').read_text());assert result['state']=='complete'
    status(state='finished',exit_code=code,result=result)
except Exception as error:
    status(state='needs_repair',error=repr(error),traceback=traceback.format_exc());raise
