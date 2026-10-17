"""Training-lifetime maintenance, at most once per600s, with a singleton lock."""
import fcntl,json,os,select,subprocess,sys,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');folder=base/'evidence_archives'
lock=(folder/'active_maintenance.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(base/'symbolic_recovery_v28'))
from nesy_vla.trace_images import collect_unreferenced_images
handles={}
for family in ['symbolic_teacher_v25','symbolic_recovery_v28']:
    for job in json.loads((base/family/'processes.json').read_text()):
        pid=job['supervisor_pid'];cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and str(base/family).encode() in cmd.read_bytes():
            try:handles[os.pidfd_open(pid)]=pid
            except ProcessLookupError:pass
assert handles,'No active training supervisor'
state={'pid':os.getpid(),'started':time.time(),'supervisor_pids':list(handles.values()),'interval_seconds':600,'cycles':[]}
status=folder/'active_maintenance.json'
def publish():
    temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status)
try:
    while handles:
        began=time.time();state['state']='archiving';publish()
        with (folder/'active_maintenance.log').open('ab') as log:
            result=subprocess.run([sys.executable,str(base/'archive_active_evidence.py')],stdout=log,stderr=subprocess.STDOUT)
        gc=collect_unreferenced_images(folder/'live_png_store')
        state['cycles'].append({'time':began,'exit_code':result.returncode,'orphan_png_links_removed':len(gc)})
        state['state']='waiting';state['next_cycle_not_before']=began+600;publish()
        while handles and time.time()<began+600:
            ready=select.select(list(handles),[],[],max(0,began+600-time.time()))[0]
            for fd in ready:os.close(fd);del handles[fd]
finally:
    for fd in handles:os.close(fd)
    state.update(state='complete',finished=time.time());publish()
