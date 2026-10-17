"""Recover the observed v56 post-evaluation retention assertion, no new run."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OLD=ROOT/'run_definition_rules_v56';SRC=ROOT/'src_definition_rules_v56'
DIAG=ROOT/'diagnostic_goal_equivalence_v57';NEXT=ROOT/'src_goal_equivalence_v57'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2))
def live(pid):
    p=Path(f'/proc/{pid}/cmdline');return bool(p.exists() and p.read_bytes())
sv=read(OLD/'supervisor.json');old_handoff=read(DIAG/'handoff.launch.json')
assert sv['state']=='needs_repair' and not live(sv['pid'])
assert not live(old_handoff['pid'])
for r in (0,1):assert not live(read(OLD/f'rank{r}.status.json')['pid'])
log=(OLD/'train.log').read_text();assert "out.name=='run_temporal_symbols_v55'" in log
commit=read(OLD/'latest.COMMITTED.json');assert commit['update']==500
assert hashlib.sha256((OLD/commit['filename']).read_bytes()).hexdigest()==commit['sha256']
development=[json.loads(s) for s in (OLD/'development.jsonl').read_text().splitlines() if s]
assert len(development)==1 and development[0]['update']==500 and development[0]['episodes']==80
gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(s)<500 for s in gpu.splitlines())
# Integration-check the repaired cleanup on this real, completed evaluation.
result=subprocess.run(['python3',str(SRC/'storage.py')],capture_output=True,text=True,check=True)
assert json.loads(result.stdout)==[], 'Only one complete evaluation; retain it all'
incident=dict(time=time.time(),reason='Copied retention path assertion still named v55; corrected to exact own run in both v56/v57 and builders',
    old_supervisor=sv,old_handoff=old_handoff,source=commit,development_preserved=development,
    checkpoint_model_optimizer_rng_preserved=True,no_completed_evaluation_repeated=True,cleanup_check=result.stdout)
write(DIAG/'v56_retention_incident.json',incident)
with (OLD/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(['python3','-u',str(SRC/'supervisor.py')],cwd=SRC,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
    if read(OLD/'supervisor.json').get('pid')==child.pid:break
    time.sleep(.1)
else:raise RuntimeError('New supervisor has not acknowledged startup; inspect process, do not relaunch')
assert live(child.pid)
with (DIAG/'handoff.log').open('ab',buffering=0) as log:
    follow=subprocess.Popen(['python3','-u',str(NEXT/'handoff.py')],cwd=NEXT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(DIAG/'handoff.launch.json',dict(pid=follow.pid,time=time.time(),source=str(NEXT),
    predecessor='run_definition_rules_v56',previous_dispatch=old_handoff))
incident.update(resumed_supervisor_pid=child.pid,handoff_pid=follow.pid,resume_update=500)
write(DIAG/'v56_retention_incident.json',incident)
print(json.dumps(incident))
