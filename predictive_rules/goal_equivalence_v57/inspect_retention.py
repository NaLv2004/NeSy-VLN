import json,os
from pathlib import Path
r=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');result=[]
for name in ('run_hybrid_replay_v54','run_temporal_symbols_v55','run_goal_aware_symbols_v52','run_contextual_symbols_v48'):
 p=r/name;sv=json.loads((p/'supervisor.json').read_text());process=Path('/proc/'+str(sv['pid'])+'/cmdline')
 if sv['state']!='finished' or (process.exists() and process.read_bytes()):continue
 best=json.loads((p/'best.COMMITTED.json').read_text());latest=json.loads((p/'latest.COMMITTED.json').read_text());keep={best['filename'],latest['filename']}
 candidates=[]
 for f in p.glob('checkpoint_*.pt'):
  c=json.loads(f.with_suffix('.COMMITTED.json').read_text());ev=p/('dev_%06d'%c['update'])
  if f.name not in keep and not ev.exists():candidates.append(dict(path=str(f),bytes=f.stat().st_size,links=f.stat().st_nlink,commit=c))
 result.append(dict(run=name,supervisor=sv,best=best,latest=latest,candidates=candidates))
print(json.dumps(result))