import argparse,json,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010/original_ppo_resume_v99')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
p=argparse.ArgumentParser();p.add_argument('--startup',action='store_true');p.add_argument('--user-requested',action='store_true');a=p.parse_args()
last=read(B/'inspection.json')
if last and time.time()-last['time']<600 and not(a.startup or a.user_requested):
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
if a.startup:
 deadline=time.monotonic()+45
 target=((read(B/'UNLIMITED_REQUEST.json') or {}).get('preserve_checkpoint') or {'epoch':70})['epoch']+1
 while ((read(B/'latest/COMMITTED') or {}).get('epoch',0)<target) and time.monotonic()<deadline:
  if any((read(B/f'rank{i}/supervisor.json') or {}).get('state')=='failed' for i in range(8)):break
  time.sleep(1)
def live(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return ''
r=dict(time=time.time(),commit=read(B/'latest/COMMITTED'),jobs=[],best=read(B/'best_score.json'),
 evaluations={f.parent.name:read(f) for f in sorted((B/'evaluations').glob('*/summary.json'))},
 early_stop=read(B/'EARLY_STOP.json'),restored=read(B/'RESTORE_VERIFIED.json'))
for j in read(B/'launcher.json')['jobs']:
 folder=B/f"rank{j['rank']}";s=read(folder/'supervisor.json');status=read(folder/'status.json') or {}
 entry=dict(rank=j['rank'],supervisor=s,trainer_live=live((s or {}).get('pid')),
  status={k:status[k] for k in ['state','update','epoch','seconds','error','traceback','symbolic_proof_max_error','ppo_replay_max_ratio_error','synchronization_error','actor_gradient_norms','completion_reason'] if k in status},startup=read(folder/'startup_verified.json'))
 if not status or status.get('state')=='failed' or (s or {}).get('state')=='failed':
  f=folder/'driver.log';entry['log_tail']=f.read_text(errors='replace')[-4000:] if f.exists() else None
 r['jobs'].append(entry)
f=B/'rank0/updates.jsonl'
if f.exists():
 records=[json.loads(l) for l in f.read_text().splitlines() if l.strip()];w=records[-10:]
 if w:
  n=sum(x['training']['episodes'] for x in w);k=sum(x['training']['successes'] for x in w)
  r['window']=dict(first_update=w[0]['update'],last_update=w[-1]['update'],episodes=n,successes=k,success_rate=k/n,
   mean_seconds=sum(x['seconds'] for x in w)/len(w),max_proof_error=max(x['symbolic_proof_max_error'] for x in w),
   max_replay_error=max(x['ppo_replay_max_ratio_error'] for x in w),max_sync_error=max(x['synchronization_error'] for x in w))
try:r['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=8)
except subprocess.TimeoutExpired:r['gpus']='GPU status query timed out; inspect trainer checkpoints separately.'
(B/'inspection.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
