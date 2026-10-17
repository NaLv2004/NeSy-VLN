import ctypes,json,os,select,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diverse_history_v70')
libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK)
assert fd>=0
watch=libc.inotify_add_watch(fd,os.fsencode(root/'episodes'),0x00000080|0x00000100)
assert watch>=0
pid=json.loads((root/'launch.json').read_text())['pid']
try:handle=os.pidfd_open(pid)
except ProcessLookupError:handle=None
paths=sorted((root/'episodes').glob('*.json'));deadline=time.monotonic()+45
while not paths and time.monotonic()<deadline:
    ready,_,_=select.select([fd]+([handle] if handle is not None else []),[],[],max(0,deadline-time.monotonic()))
    if fd in ready:
        os.read(fd,65536);paths=sorted((root/'episodes').glob('*.json'))
    if handle is None or handle in ready:break
os.close(fd)
if handle is not None:os.close(handle)
meta=json.loads(paths[0].read_text()) if paths else None
if meta:meta={k:v for k,v in meta.items() if k not in ('steps','programs')}
record=dict(time=time.time(),pid=pid,committed_files=len(paths),first_committed=meta,
    collection_state=json.loads((root/'status.json').read_text()))
(root/'first_data_observation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
