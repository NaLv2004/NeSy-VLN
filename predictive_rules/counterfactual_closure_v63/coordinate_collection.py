import json,os,select,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'counterfactual_closure_v63';SOURCE=ROOT/'src_counterfactual_closure_v63'
PY='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def write(row):
    row.update(pid=os.getpid(),time=time.time());p=OUT/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
processes=[];fds={}
try:
    for rank in (0,1):
        log=(OUT/f'collector_rank{rank}.log').open('ab',buffering=0)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(6+rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        p=subprocess.Popen([PY,'-u',str(SOURCE/'collect_closure.py'),'--out',str(OUT),'--checkpoint',str(OUT/'source.pt'),
            '--tasks','600','--workers','24','--rank',str(rank)],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);fds[os.pidfd_open(p.pid)]=p
    write(dict(state='collecting',workers=[p.pid for p in processes],tasks=600,started=time.time()))
    while fds:
        ready,_,_=select.select(list(fds),[],[])
        for fd in ready:
            p=fds.pop(fd);os.close(fd);code=p.wait()
            if code:
                for other in processes:
                    if other.poll() is None:other.terminate()
                raise RuntimeError(dict(failed_pid=p.pid,exit_code=code))
    states=[json.loads((OUT/f'status_rank{r}.json').read_text()) for r in (0,1)]
    assert all(s['state']=='complete' for s in states)
    write(dict(state='complete',workers=[p.pid for p in processes],exit_codes=[p.returncode for p in processes],
        completed=sum(s['completed'] for s in states),positions=sum(s['positions'] for s in states),
        one_step_outcomes=sum(s['positions'] for s in states)*17,two_step_outcomes=sum(s['positions'] for s in states)*32))
except Exception as e:
    write(dict(state='needs_repair',workers=[p.pid for p in processes],error=repr(e),traceback=traceback.format_exc()));raise
finally:
    for fd in fds:os.close(fd)
