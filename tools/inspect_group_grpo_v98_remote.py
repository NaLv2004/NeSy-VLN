import argparse,json,subprocess,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010/group_grpo_v98')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
p=argparse.ArgumentParser();p.add_argument('--startup',action='store_true');p.add_argument('--user-requested',action='store_true');a=p.parse_args()
last=read(B/'inspection.json')
if last and time.time()-last['time']<600 and not (a.startup or a.user_requested):
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
def live(pid):
 f=Path(f'/proc/{pid}/cmdline');return f.read_bytes().replace(b'\0',b' ').decode() if pid and f.exists() else None
r={'time':time.time(),'commit':read(B/'latest/COMMITTED'),'jobs':[],'source':read(B/'SOURCE.json')['commit'],
   'best':read(B/'best_score.json'),'evaluations':{f.parent.name:read(f) for f in sorted((B/'evaluations').glob('*/summary.json'))}}
for j in read(B/'launcher.json')['jobs']:
 folder=B/f"rank{j['rank']}";s=read(folder/'supervisor.json');status=read(folder/'status.json')
 item=dict(rank=j['rank'],supervisor=s,trainer_live=live((s or {}).get('pid')),supervisor_live=live(j['supervisor_pid']),
  status={k:v for k,v in (status or {}).items() if k in ['state','time','update','epoch','error','traceback','seconds','zero_variance_groups','accepted_optimizer_epochs','next_actor_lr','symbolic_proof_max_error','grpo_replay_max_ratio_error','synchronization_error','peak_gpu_allocated_gib']},
  startup=read(folder/'startup_verified.json'))
 if not status or status.get('state')=='failed':
  f=folder/'driver.log';item['log_tail']=f.read_text(errors='replace')[-6000:] if f.exists() else None
 r['jobs'].append(item)
f=B/'rank0/updates.jsonl'
if f.exists():
 records=[json.loads(line) for line in f.read_text().splitlines() if line.strip()];recent=records[-10:]
 if recent:
  count=sum(x['training']['episodes'] for x in recent);success=sum(x['training']['successes'] for x in recent)
  r['window']=dict(first_update=recent[0]['update'],last_update=recent[-1]['update'],episodes=count,successes=success,
   success_rate=success/count,mean_seconds=sum(x['seconds'] for x in recent)/len(recent),
   mean_return_std=sum(g['return_std'] for x in recent for g in x['group_audits'])/(24*len(recent)),
   zero_variance_groups=sum(x['zero_variance_groups'] for x in recent),groups=24*len(recent),
   reference_kl_mean=sum(sum(x['reference_kls'])/len(x['reference_kls']) for x in recent)/len(recent),
   first_step_same_image_all=all(g['initial_images_identical'] for x in recent for g in x['group_audits']),
   max_proof_error=max(x['symbolic_proof_max_error'] for x in recent),
   max_replay_error=max(x['grpo_replay_max_ratio_error'] for x in recent),
   max_sync_error=max(x['synchronization_error'] for x in recent))
  r['latest_training']=recent[-1]['training'];r['latest_groups']=recent[-1]['group_audits']
r['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
(B/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
