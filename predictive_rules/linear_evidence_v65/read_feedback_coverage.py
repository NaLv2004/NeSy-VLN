import json,collections
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
rows=[json.loads(f.read_text()) for f in (root/'goal_discrimination_v64/episodes').glob('*.json')]
print(json.dumps(dict(tasks=len(rows),positive_episodes=dict(collections.Counter(r['category'] for r in rows if r['positive_stop_outcomes']>0)),
    positive_frames=dict((c,sum(r['positive_stop_outcomes'] for r in rows if r['category']==c)) for c in ('chair','table')))))
