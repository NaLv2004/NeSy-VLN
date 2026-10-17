import json,collections,time,statistics
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010/symbolic_joint_v9');run=base/'runs/joint'
rows=[]
for path in sorted((run/'traces').glob('update_*/*/trajectory.json')):
    try:t=json.loads(path.read_text())
    except json.JSONDecodeError:continue
    steps=t['steps']
    if not steps:continue
    rewards=[s['reward'] for s in steps]
    rows.append({'path':str(path),'episode':t['episode_id'],'group':t['distance_group'],'success':bool(t['metrics'].get('success')),
        'steps':len(steps),'stop':steps[-1]['action']==0,'return':sum(rewards),
        'discounted_return':sum((.99**i)*r for i,r in enumerate(rewards)),
        'collision_steps':sum(s['collision'] for s in steps),'actions':[s['action'] for s in steps]})
groups={}
for name,predicate in [('successful',lambda x:x['success']),('failed_stop',lambda x:not x['success'] and x['stop']),('failed_horizon',lambda x:not x['success'] and not x['stop'])]:
    selected=[x for x in rows if predicate(x)]
    groups[name]={'episodes':len(selected),'mean_steps':statistics.mean(x['steps'] for x in selected) if selected else None,
        'mean_return':statistics.mean(x['return'] for x in selected) if selected else None,
        'discounted_returns_below_false_stop_reference':sum(x['discounted_return']<-12.04 for x in selected)}
result={'time':time.time(),'trace_sampling':'One rotating episode per rank/update, not all training rollouts',
    'false_stop_reference':-12.04,'groups':groups,'rows':rows,
    'qualification':'This compares recorded returns to the false-STOP reward, not an executed counterfactual from each initial pose.'}
(base/'training_return_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({'groups':groups,'sampled_episodes':len(rows)},indent=2))
