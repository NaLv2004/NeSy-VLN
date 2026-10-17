"""Kernel exit notification, not periodic status polling."""
import json,os,select,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
handles={};already=[]
for name in ('stable_rules_v75','expanded_history_v76'):
    out=ROOT/('run_'+name);row=json.loads((out/'training_launch.json').read_text());pid=row['pid']
    try:handles[os.pidfd_open(pid)]=name
    except ProcessLookupError:already.append(name)
ready=[]
if not already and handles:ready=select.select(list(handles),[],[],45)[0]
finished=already+[handles[fd] for fd in ready]
for fd in handles:os.close(fd)
results={}
for name in finished:
    out=ROOT/('run_'+name)
    results[name]={fn:json.loads((out/fn).read_text()) for fn in ('supervisor.json','result.json','best.COMMITTED.json') if (out/fn).exists()}
print(json.dumps(dict(time=time.time(),finished=results,event='pidfd_exit' if finished else 'bounded_wait_timeout')))
