import json,statistics
from pathlib import Path
p=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_induced_effects_v39/rank0.jsonl')
rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
for lo,hi in [(801,1000),(1001,1200)]:
 x=[r['gradient_norm'] for r in rows if lo<=r['update']<=hi]
 print(json.dumps({'updates':[lo,hi],'n':len(x),'median':statistics.median(x),'max':max(x),'over1000':sum(v>1000 for v in x)}))