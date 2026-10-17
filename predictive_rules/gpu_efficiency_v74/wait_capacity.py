"""Watch kernel file events for capacity measurements; no interval polling."""
import ctypes,json,os,select,struct,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
names=['stable_rules_v75','expanded_history_v76'];lib=ctypes.CDLL(None,use_errno=True)
fd=lib.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);assert fd>=0
for name in names:assert lib.inotify_add_watch(fd,os.fsencode(ROOT/('run_'+name)),0x80|0x8)>=0
deadline=time.monotonic()+45
try:
    while not all((ROOT/('run_'+n)/'capacity.json').exists() for n in names):
        left=deadline-time.monotonic()
        if left<=0 or not select.select([fd],[],[],left)[0]:break
        os.read(fd,1<<18)
    result={}
    for n in names:
        out=ROOT/('run_'+n);result[n]={fn:json.loads((out/fn).read_text()) for fn in ('capacity.json','status_rank0.json','supervisor.json') if (out/fn).exists()}
    print(json.dumps(dict(time=time.time(),runs=result)))
finally:os.close(fd)
