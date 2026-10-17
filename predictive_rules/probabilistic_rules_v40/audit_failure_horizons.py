from collections import Counter
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]);rows=[json.loads(p.read_text()) for p in sorted(root.glob('*/trace.json'))]
unsupported={r['episode'] for r in json.loads((root/'goal_support_audit.json').read_text())['program_rows'] if r['stopped_before_next_program'] and not r['any_stop_preimage']}
histogram=Counter();unsupported_hist=Counter();failures=[]
for row in rows:
    if not row.get('planning_failed'):continue
    proof=row['failed_proof'];h=proof['horizon'];histogram[h]+=1
    if row['episode']['episode_id'] in unsupported:unsupported_hist[h]+=1
    failures.append(dict(episode=row['episode']['episode_id'],steps=len(row['steps']),remaining_horizon=h,
        domain=row['episode']['domain'],distance=row['episode']['distance_group'],
        unsupported_goal=row['episode']['episode_id'] in unsupported))
result=dict(episodes=len(rows),failure_remaining_horizons=dict(histogram),unsupported_goal_horizons=dict(unsupported_hist),failures=failures)
(root/'failure_horizon_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='failures'}))
