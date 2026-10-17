import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'predictive_rules/goal_progress.json';d=json.loads(p.read_text())
d['rule_inverse_trial'].update(development_evaluation='complete',development_successes=0,development_episodes=80,
    development_planning_failures=41,promoted=False,state='bounded_trial_completed')
d['pending']=[item for item in d['pending'] if not item.startswith('Read the bounded')]
p.write_text(json.dumps(d,indent=2),encoding='utf-8')
