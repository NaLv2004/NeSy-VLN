import shutil,json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12'
assert not root.exists();root.mkdir()
for name in ['nesy_vla','scripts','tests']:
    shutil.copytree(base/'symbolic_reward_v11'/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'experiment_pending.json').write_text(json.dumps({'time':time.time(),
    'state':'source_preparation_only','training_started':False,
    'diagnostic':str(base/'paired_goal_probe_v12'),'data':str(base/'data/paired_goals_v12'),
    'reason':'Wait for goal, image and predicate interventions before choosing a full training experiment'},indent=2))
print(root)
