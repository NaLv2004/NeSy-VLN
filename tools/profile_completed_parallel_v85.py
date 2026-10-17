"""Compare recorded completed update timings; never reruns a rollout."""
import json,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');result={'time':time.time(),'runs':{}}
for name in ['symbolic_ppo8_v82','parallel_ppo_v84','parallel_ppo_v85']:
 p=B/name/'rank0/updates.jsonl'
 if not p.exists():continue
 rows=[json.loads(s) for s in p.read_text().splitlines()]
 # Exclude first cold-start update and use at most ten recent completed updates.
 steady=rows[1:][-10:]
 if not steady:continue
 n=sum(r['training']['episodes'] for r in steady);seconds=sum(r['seconds'] for r in steady)
 x={'first_update':steady[0]['update'],'last_update':steady[-1]['update'],'updates':len(steady),
  'episodes':n,'mean_update_seconds':seconds/len(steady),'rollouts_per_second':n/seconds,
  'first_cold_update_seconds':rows[0]['seconds'],'timings':[{k:r[k] for k in ['update','seconds','collection_timing','optimization_and_commit_seconds','peak_gpu_allocated_gib'] if k in r} for r in steady]}
 result['runs'][name]=x
(B/'parallel_performance_v85.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
