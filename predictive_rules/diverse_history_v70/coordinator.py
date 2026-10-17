import json,os,select,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diverse_history_v70';SOURCE=ROOT/'src_diverse_history_v70'
PY='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
def write(row):
    row.update(pid=os.getpid(),time=time.time());p=OUT/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
processes=[];fds={}
try:
    for rank in (0,1):
        with (OUT/f'collector_rank{rank}.log').open('ab',buffering=0) as log:
            p=subprocess.Popen([PY,'-u',str(SOURCE/'collect.py'),'--rank',str(rank)],cwd=SOURCE,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(6+rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);fds[os.pidfd_open(p.pid)]=p
    write(dict(state='collecting',workers=[p.pid for p in processes],tasks=1200,started=time.time()))
    while fds:
        ready,_,_=select.select(list(fds),[],[])
        for fd in ready:
            p=fds.pop(fd);os.close(fd);code=p.wait()
            if code:
                for other in processes:
                    if other.poll() is None:other.terminate()
                raise RuntimeError(dict(failed_pid=p.pid,exit_code=code))
    rows=[json.loads((OUT/f'status_rank{r}.json').read_text()) for r in (0,1)]
    assert all(r['state']=='complete' for r in rows)
    write(dict(state='complete',workers=[p.pid for p in processes],exit_codes=[p.returncode for p in processes],
        completed=sum(r['completed'] for r in rows),frames=sum(r['frames'] for r in rows),outcomes=sum(r['outcomes'] for r in rows),successes=sum(r['successes'] for r in rows)))
except Exception as error:
    write(dict(state='needs_repair',workers=[p.pid for p in processes],error=repr(error),traceback=traceback.format_exc()));raise
finally:
    for fd in fds:os.close(fd)
