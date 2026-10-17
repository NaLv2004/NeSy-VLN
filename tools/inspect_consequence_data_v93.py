"""Inspect existing training-only execution records, without touching the live run."""
import collections, hashlib, json
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010');R=Path('/home/user/xjy/strips_wm_navigation_20261009')
read=lambda p:json.loads(p.read_text())
rows=read(B/'perceptual_sequences_v73/tail_1e4/training_rows.json')
paths=[R/'data/search_cost_supervision_v3/execution_outcomes.json',B/'data/onpolicy_recovery_v48/execution_outcomes.json']
report={'training_rows':len(rows),'first_rows':rows[:2],'rows_by_episode':len(set(r['episode_id'] for r in rows)),'outcomes':[]}
for path in paths:
 if not path.exists():report['outcomes'].append({'path':str(path),'missing':True});continue
 records=read(path);report['outcomes'].append({'path':str(path),'rows':len(records),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'first':records[0]})
report['source_commit']=read(B/'perceptual_sequences_v73/tail_1e4/TOKENS_COMMITTED')
report['extra_commit']=read(B/'data/onpolicy_recovery_v48/COMPLETE')
print(json.dumps(report,indent=2))
