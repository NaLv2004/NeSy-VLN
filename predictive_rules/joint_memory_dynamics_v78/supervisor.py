import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_joint_memory_dynamics_v78';OUT=ROOT/'run_joint_memory_dynamics_v78'
PY='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def status(**row):
    row.update(pid=os.getpid(),time=time.time());p=OUT/'supervisor.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
try:
    with (OUT/'training.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([PY,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(SOURCE/'train_joint.py')],
            cwd=SOURCE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6,7',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='2'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status(state='training',child=p.pid);code=p.wait()
    if code:raise RuntimeError(dict(exit_code=code))
    result=json.loads((OUT/'result.json').read_text());assert result['state']=='complete'
    status(state='finished',exit_code=code)
except Exception as error:
    status(state='needs_repair',error=repr(error),traceback=traceback.format_exc());raise
