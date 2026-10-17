"""Apply the user's continuous-training instruction, preserving committed state."""
import fcntl,hashlib,json,os,signal,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010/original_ppo_resume_v99')
read=lambda p:json.loads(p.read_text())
lock=(B/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (B/'UNLIMITED_REQUEST.json').exists(), 'Inspect existing transition instead of repeating it'
p=B/'src/train.py';old_text=p.read_text();text=old_text
text=text.replace('import argparse,datetime,fcntl,gc,hashlib,json,os,random,sys,time,traceback','import argparse,datetime,fcntl,gc,hashlib,itertools,json,os,random,sys,time,traceback')
assert "'updates':args.updates" in text
text=text.replace("'updates':args.updates","'updates':None")
text=text.replace("'early_stop_no_improvement_validations':3","'early_stop_no_improvement_validations':None")
anchor="if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest"
replacement="""def compatible_manifest(saved,current):
 return {k:v for k,v in saved.items() if k not in ('updates','early_stop_no_improvement_validations')}=={k:v for k,v in current.items() if k not in ('updates','early_stop_no_improvement_validations')}
if (out/'manifest.json').exists():
 assert compatible_manifest(read(out/'manifest.json'),manifest)
 if rank==0:atomic(out/'manifest.json',manifest)"""
assert anchor in text;text=text.replace(anchor,replacement)
text=text.replace("assert saved['manifest']==manifest","assert compatible_manifest(saved['manifest'],manifest)")
anchor=" stale_validations=0;best_seen=read(out/'best_score.json')['success_rate'];last_completed=begin;completion_reason='update_limit'"
assert anchor in text;text=text.replace(anchor,'')
anchor='  for update in range(begin+1,args.updates+1):'
assert anchor in text;text=text.replace(anchor,'  for update in itertools.count(begin+1):')
anchor="""   last_completed=update
   if update%10==0:
    summary=evaluate(update)
    if summary['success_rate']>best_seen:
     best_seen=summary['success_rate'];stale_validations=0
    else:stale_validations+=1
    if stale_validations>=3:
     completion_reason='three_validations_without_improvement'
     if rank==0:atomic(out/'EARLY_STOP.json',{'time':time.time(),'update':update,'best_success_rate':best_seen,'latest_success_rate':summary['success_rate'],'no_restart':True})
     break
 progress(state='complete',update=last_completed,completion_reason=completion_reason)"""
assert anchor in text;text=text.replace(anchor,"   if update%10==0:evaluate(update)")
assert 'stale_validations' not in text and 'itertools.count(begin+1)' in text
compile(text,str(p),'exec')
jobs=read(B/'launcher.json')['jobs'];members=[]
for j in jobs:
 s=read(B/f"rank{j['rank']}"/'supervisor.json');sid=j['supervisor_pid']
 for pid,expected in [(sid,str(B/'supervise.py')),(s['pid'],str(B/'src/train.py'))]:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert expected in cmd and os.getpgid(pid)==sid
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  stat=(d/'stat').read_text().rsplit(')',1)[1].split()
  if int(stat[2]) in [j['supervisor_pid'] for j in jobs] and stat[0]!='Z':
   members.append(dict(pid=int(d.name),start=stat[19]))
 except FileNotFoundError:pass
frozen=[]
try:
 for j in jobs:os.killpg(j['supervisor_pid'],signal.SIGSTOP);frozen.append(j['supervisor_pid'])
 saved=B/'latest' if (B/'latest/COMMITTED').exists() else B/'source_snapshot'
 commit=read(saved/'COMMITTED');assert hashlib.sha256((saved/commit['checkpoint_file']).read_bytes()).hexdigest()==commit['checkpoint_sha256']
 (B/'src/train_before_unlimited.py').write_text(old_text);p.write_text(text)
 r=dict(time=time.time(),user_requested=True,maximum_updates=None,early_stop=False,
        preserve_checkpoint=commit,resume_from=str(saved),old_jobs=jobs,old_processes=members,
        save_every_update=True,evaluation_every=10,status='old_processes_stopping')
 (B/'UNLIMITED_REQUEST.json').write_text(json.dumps(r,indent=2))
except BaseException:
 for pgid in frozen:
  try:os.killpg(pgid,signal.SIGCONT)
  except ProcessLookupError:pass
 raise
for pgid in frozen:
 try:os.killpg(pgid,signal.SIGTERM);os.killpg(pgid,signal.SIGCONT)
 except ProcessLookupError:pass
print(json.dumps({k:v for k,v in r.items() if k not in ['old_jobs','old_processes']},indent=2))
