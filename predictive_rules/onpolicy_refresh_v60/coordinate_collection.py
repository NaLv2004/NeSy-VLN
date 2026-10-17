import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'onpolicy_successors_v60';SOURCE=ROOT/'src_onpolicy_refresh_v60'
PYTHON='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
OUT.mkdir(exist_ok=True)
def save(row):
    path=OUT/'status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(path)
processes=[]
for rank in (0,1):
    log=(OUT/f'collector_rank{rank}.log').open('ab',buffering=0)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(6+rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    p=subprocess.Popen([PYTHON,'-u',str(SOURCE/'collect_onpolicy.py'),'--out',str(OUT),
        '--checkpoint',str(OUT/'source.pt'),'--tasks','120','--workers','4','--rank',str(rank)],
        cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    processes.append(p)
save(dict(state='collecting',pid=os.getpid(),workers=[p.pid for p in processes],started=time.time()))
codes=[p.wait() for p in processes]
states=[json.loads((OUT/f'status_rank{r}.json').read_text()) if (OUT/f'status_rank{r}.json').exists() else {} for r in (0,1)]
ok=codes==[0,0] and all(s.get('state')=='complete' for s in states)
save(dict(state='complete' if ok else 'needs_repair',pid=os.getpid(),exit_codes=codes,
    workers=[p.pid for p in processes],time=time.time(),completed=sum(s.get('completed',0) for s in states),
    positions=sum(s.get('positions',0) for s in states),executed_actions=sum(s.get('executed_actions',0) for s in states)))
