import json,os,select,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_ensemble_rules_v61';SOURCE=ROOT/'src_ensemble_rules_v61'
PREVIOUS=ROOT/'run_onpolicy_refresh_v60'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def write(x):
    x.update(pid=os.getpid(),time=time.time());p=OUT/'evaluation.status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
    sv=json.loads((PREVIOUS/'supervisor.json').read_text());pid=sv['pid'];p=Path(f'/proc/{pid}/cmdline')
    if p.exists() and p.read_bytes():
        assert b'src_onpolicy_refresh_v60_training/supervisor.py' in p.read_bytes()
        fd=os.pidfd_open(pid);write(dict(state='waiting_training_exit',training_pid=pid))
        try:select.select([fd],[],[])
        finally:os.close(fd)
    sv=json.loads((PREVIOUS/'supervisor.json').read_text());assert sv['state']=='finished' and sv['exit_code']==0
    assert not subprocess.check_output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert shutil.disk_usage(ROOT).free>350_000_000
    assert json.loads((OUT/'status.json').read_text())['state']=='complete'
    assert not (OUT/'dev_full80').exists(),'Do not repeat a checkpoint evaluation'
    write(dict(state='checking_export_contract'))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (OUT/'export_contract.log').open('ab',buffering=0) as log:
        subprocess.run([PYTHON,'-u',str(SOURCE/'check_export.py')],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    workers=[]
    for rank in (0,1):
        log=(OUT/f'eval_rank{rank}.log').open('ab',buffering=0)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(6+rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        p=subprocess.Popen([PYTHON,'-u',str(SOURCE/'run_evaluation.py'),'--rank',str(rank)],cwd=SOURCE,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True);workers.append(p)
    write(dict(state='evaluating_full80',workers=[p.pid for p in workers]))
    codes=[p.wait() for p in workers];assert codes==[0,0],codes
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    subprocess.run([PYTHON,'-u',str(SOURCE/'finish_evaluation.py')],cwd=SOURCE,env=env,check=True)
    write(dict(state='complete',exit_codes=codes,workers=[p.pid for p in workers]))
try:main()
except Exception as e:write(dict(state='needs_repair',error=repr(e),traceback=traceback.format_exc()));raise
