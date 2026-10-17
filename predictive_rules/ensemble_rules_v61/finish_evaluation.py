import json,time
from pathlib import Path
from evaluate import summarize
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'diagnostic_ensemble_rules_v61'
folder=OUT/'dev_full80';rows=[]
for rank in (0,1):rows+=json.loads((folder/f'rank{rank}.episodes.json').read_text())
assert len(rows)==80 and len({r['episode_id'] for r in rows})==80
baseline=ROOT/'run_goal_equivalence_v57/dev_001000';old={}
for p in baseline.glob('*/trace.json'):
    r=json.loads(p.read_text());old[r['episode']['episode_id']]=bool(r['metrics'].get('success'))
assert set(old)=={r['episode_id'] for r in rows}
result=summarize(rows);result.update(time=time.time(),source_update=1000,baseline_successes=sum(old.values()),
    gained=sum(r['success'] and not old[r['episode_id']] for r in rows),
    lost=sum(not r['success'] and old[r['episode_id']] for r in rows),
    same_original_full80=True,baseline_not_rerun=True,finalfixed80_used=False)
(folder/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
