"""Stop only the verified stuck v76 evaluation and launch sealed-weight eval."""
import os,json,time,signal,subprocess,shutil
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OLD=ROOT/'src_expanded_history_v76';NEW=ROOT/'src_expanded_history_v76_evalfix'
OUT=ROOT/'run_expanded_history_v76'
def command(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
if (OUT/'evaluation_repair.json').exists():raise RuntimeError('Repair already launched; inspect actual handle')
roots=[]
for fn in ('supervisor.json','status_rank0.json','status_rank1.json'):
    record=json.loads((OUT/fn).read_text());pid=record['pid']
    if command(pid):
        assert str(OLD) in command(pid),(pid,command(pid));roots.append(pid)
parents={}
for p in Path('/proc').iterdir():
    if p.name.isdigit():
        try:parents[int(p.name)]=int(p.joinpath('stat').read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError):pass
owned=set(roots)
while True:
    children={pid for pid,ppid in parents.items() if ppid in owned}
    if children<=owned:break
    owned|=children
# Open stable process handles before sending a signal; avoid PID reuse.
handles=[]
for pid in sorted(owned):
    try:handles.append((pid,os.pidfd_open(pid)))
    except ProcessLookupError:pass
for pid,fd in handles:
    try:signal.pidfd_send_signal(fd,signal.SIGTERM)
    except ProcessLookupError:pass
import select
pending=[fd for pid,fd in handles];deadline=time.time()+20
while pending and time.time()<deadline:
    ended,_,_=select.select(pending,[],[],1);pending=[fd for fd in pending if fd not in ended]
for fd in pending:
    try:signal.pidfd_send_signal(fd,signal.SIGKILL)
    except ProcessLookupError:pass
for pid,fd in handles:os.close(fd)
assert not any(command(pid) for pid in roots),'Old training ranks still active'
shutil.copytree(OLD,NEW)
for name in ('evaluate.py','resume_evaluation.py'):
    shutil.copy2(ROOT/'repair_v76_staging'/name,NEW/name)
py='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python'
with (OUT/'evaluation_repair.log').open('ab',buffering=0) as log:
    p=subprocess.Popen([py,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(NEW/'resume_evaluation.py')],
        cwd=NEW,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='2'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,time=time.time(),state='evaluating_repaired',terminated_exact_pids=sorted(owned),source=str(NEW),
    checkpoints_preserved=True,optimizer_updates=0,reason='Repeated list append during iteration in generated evaluation code')
(OUT/'evaluation_repair.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
for name in ('expanded_dynamics_v71','stable_rules_v75'):
    p=ROOT/('run_'+name)/'result.json'
    if p.exists():print(name,p.read_text())
