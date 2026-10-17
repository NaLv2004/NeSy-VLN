"""Replace only the owned v63 collectors, resuming all committed episodes."""
import json,os,select,signal,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
SOURCE=ROOT/'src_counterfactual_closure_v63';DATA=ROOT/'counterfactual_closure_v63';RUN=ROOT/'run_counterfactual_closure_v63'
read=lambda p:json.loads(p.read_text())
launch=read(DATA/'launcher.json');status=read(DATA/'status.json');handoff=read(RUN/'launcher.json');sv=read(RUN/'supervisor.json')
assert status['state']=='collecting' and sv['state']=='waiting_collection',(status,sv)
assert not (RUN/'latest.COMMITTED.json').exists(),'Training has started; do not replace active fitter'
expected={launch['pid']:'coordinate_collection.py',handoff['pid']:'continue_after_collection.py'}
expected.update({pid:'collect_closure.py' for pid in status['workers']})
for pid,script in expected.items():
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(SOURCE/script) in cmd,(pid,cmd)
all_processes={}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        st=(proc/'stat').read_text().rsplit(')',1)[1].split()
        all_processes[int(proc.name)]=(int(st[1]),(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (OSError,ValueError):pass
owned=set(expected)
while True:
    extra={pid for pid,(parent,cmd) in all_processes.items() if parent in owned}-owned
    if not extra:break
    for pid in extra:assert str(SOURCE/'closure_worker.py') in all_processes[pid][1],all_processes[pid]
    owned.update(extra)
stamp=str(int(time.time()));archive=DATA/('optimization_'+stamp);archive.mkdir()
before={f:read(DATA/f) for f in ('launcher.json','status.json','status_rank0.json','status_rank1.json')}
before['handoff']=handoff;before['supervisor']=sv
(archive/'before.json').write_text(json.dumps(before,indent=2))
# First stop the waiting handoff, so a deliberately drained collector cannot
# be mistaken for a successful600-task completion and launch training early.
handles={}
for pid in owned:
    try:handles[pid]=os.pidfd_open(pid)
    except ProcessLookupError:pass
order=[handoff['pid'],launch['pid'],*status['workers'],*sorted(owned-set(expected))]
for pid in order:
    if pid in handles:
        try:signal.pidfd_send_signal(handles[pid],signal.SIGTERM)
        except ProcessLookupError:pass
pending=set(handles.values());deadline=time.monotonic()+40
while pending and time.monotonic()<deadline:
    ready,_,_=select.select(list(pending),[],[],max(0,deadline-time.monotonic()))
    pending-=set(ready)
if pending:
    for fd in pending:
        try:signal.pidfd_send_signal(fd,signal.SIGKILL)
        except ProcessLookupError:pass
    ready,_,_=select.select(list(pending),[],[],10);pending-=set(ready)
assert not pending,'Inspect exact remaining process handles before restarting'
for fd in handles.values():os.close(fd)
# Promote staged Python files only after old interpreters have stopped.
for name in ('collect_closure.py','coordinate_collection.py'):
    (SOURCE/name).replace(archive/name)
    (SOURCE/(name+'.next')).replace(SOURCE/name)
subprocess.run(['python3','-m','compileall','-q',str(SOURCE)],check=True)
for folder in (DATA,RUN):
    (folder/'launcher.json').replace(archive/(folder.name+'_launcher.json'))
with (DATA/'coordinator.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['python3','-u',str(SOURCE/'coordinate_collection.py')],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
new=dict(launch,pid=p.pid,time=time.time(),previous_pid=launch['pid'],workers_per_gpu=24,
    change='Concurrent pure NumPy planning and24simulators/GPU; original single-image numerics retained; committed episodes resumed')
(DATA/'launcher.json').write_text(json.dumps(new,indent=2))
with (RUN/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(SOURCE/'continue_after_collection.py')],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
new_handoff=dict(handoff,pid=child.pid,time=time.time(),wait_for_collection_pid=p.pid,previous_pid=handoff['pid'])
(RUN/'launcher.json').write_text(json.dumps(new_handoff,indent=2))
result=dict(time=time.time(),owned_stopped=sorted(owned),new_collector=new,new_handoff=new_handoff,
    committed_episode_files=len(list((DATA/'episodes').glob('*.json'))),other_gpus_untouched=True,
    scene_split_unchanged=True,encoding_and_symbolic_rules_unchanged=True)
(archive/'result.json').write_text(json.dumps(result,indent=2));(DATA/'parallel_optimization.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
