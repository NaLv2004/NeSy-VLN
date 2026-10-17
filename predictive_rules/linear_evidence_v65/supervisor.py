import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_linear_evidence_v65';OUT=ROOT/'run_linear_evidence_v65'
PY='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def status(**row):
    row.update(pid=os.getpid(),time=time.time());p=OUT/'supervisor.json';temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(row,indent=2));temp.replace(p)
try:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='2')
    with (OUT/'contract.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([PY,'-u',str(SOURCE/'check_contract.py')],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(state='checking_contract',child=p.pid);code=p.wait()
    if code:raise RuntimeError(dict(stage='contract',exit_code=code))
    env['CUDA_VISIBLE_DEVICES']='6,7'
    with (OUT/'training.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([PY,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(SOURCE/'train_evidence.py')],
            cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(state='training',child=p.pid);code=p.wait()
    if code:raise RuntimeError(dict(stage='training',exit_code=code))
    result=json.loads((OUT/'result.json').read_text());assert result['state']=='complete'
    status(state='finished',exit_code=code,result=result)
except Exception as error:
    status(state='needs_repair',error=repr(error),traceback=traceback.format_exc());raise
