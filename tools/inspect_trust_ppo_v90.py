import argparse,json,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/trust_ppo_v90')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
last=read(b/'inspection.json')
parser=argparse.ArgumentParser();parser.add_argument('--user-requested',action='store_true');args=parser.parse_args()
if last and time.time()-last['time']<600 and not args.user_requested:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if pid and p.exists() else None
jobs=[]
for j in read(b/'launcher.json')['jobs']:
 folder=b/f"rank{j['rank']}";s=read(folder/'supervisor.json');status=read(folder/'status.json')
 row=dict(j,supervisor=s,trainer_live=live((s or {}).get('pid')),supervisor_live=live(j['supervisor_pid']),status=status)
 if not status or status.get('state')=='failed':row['tail']=(folder/'driver.log').read_text(errors='replace')[-4500:]
 jobs.append(row)
r={'time':time.time(),'user_requested':args.user_requested,'jobs':jobs,'commit':read(b/'latest/COMMITTED'),'best':read(b/'best_score.json'),
   'evaluations':{p.parent.name:read(p) for p in sorted((b/'evaluations').glob('*/summary.json'))},
   'free_gib':shutil.disk_usage(b).free/2**30,
   'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
updates=b/'rank0/updates.jsonl'
if updates.exists():
 records=[json.loads(line) for line in updates.read_text().splitlines() if line.strip()]
 recent=records[-10:]
 if recent:
  episodes=sum(x['training']['episodes'] for x in recent)
  actions=sum(x['global_actions'] for x in recent)
  r['recent_training_window']={'first_update':recent[0]['update'],'last_update':recent[-1]['update'],
   'episodes':episodes,'successes':sum(x['training']['successes'] for x in recent),
   'success_rate':sum(x['training']['successes'] for x in recent)/episodes,
   'collision_action_rate':sum(x['training']['collision_action_rate']*x['global_actions'] for x in recent)/actions,
   'mean_update_seconds':sum(x['seconds'] for x in recent)/len(recent),
   'rollouts_per_second':episodes/sum(x['seconds'] for x in recent),
   'max_ratio_error':max(x['ppo_replay_max_ratio_error'] for x in recent),
   'max_rule_proof_error':max(x['symbolic_proof_max_error'] for x in recent),
   'max_parameter_sync_error':max(x['synchronization_error'] for x in recent),
   'qualification':'Randomly sampled training rollouts. Time excludes periodic development evaluation.'}
if updates.exists() and recent:
 r['trust_region_recent']={'accepted_optimizer_epochs':sum(x['accepted_optimizer_epochs'] for x in recent),
  'updates_without_accepted_epoch':sum(x['accepted_optimizer_epochs']==0 for x in recent),
  'rejected_epochs':sum(not e['accepted'] for x in recent for e in x['trust_region']),
  'maximum_accepted_kl':max((e['mean_kl'] for x in recent for e in x['trust_region'] if e['accepted']),default=None),
  'next_actor_lr':recent[-1]['next_actor_lr']}
(b/'inspection.json').write_text(json.dumps(r,indent=2))
for j in jobs:
 if j['status']:j['status']={k:v for k,v in j['status'].items() if k in ['pid','state','rank','time','update','epoch','seconds','ppo_replay_max_ratio_error','symbolic_proof_max_error','synchronization_error','global_actions','error','traceback','actions','peak_gpu_allocated_gib','collection_timing','optimization_and_commit_seconds','accepted_optimizer_epochs','next_actor_lr','trust_region']}
print(json.dumps(r,indent=2))
