import json,time
from pathlib import Path
B=Path('/home/user/xjy/nesy_vla_20261010')
read=lambda p:json.loads(p.read_text())
r={'time':time.time(),'user_requested':True,'evaluations':[]}
for name in ['symbolic_ppo8_v82','parallel_ppo_v84','parallel_ppo_v85']:
 run=B/name
 for p in sorted((run/'evaluations').glob('*/summary.json')):
  e=read(p)
  r['evaluations'].append(dict(run=name,update=int(p.parent.name.split('_')[-1]),episodes=e['episodes'],successes=e['successes'],
    collision_action_rate=e['collision_action_rate']))
run=B/'parallel_ppo_v85';r['latest_commit']=read(run/'latest/COMMITTED')
r['jobs']=[]
for j in read(run/'launcher.json')['jobs']:
 s=read(run/f'rank{j["rank"]}/supervisor.json');p=Path(f'/proc/{s["pid"]}/cmdline')
 status=read(run/f'rank{j["rank"]}/status.json')
 r['jobs'].append(dict(rank=j['rank'],pid=s['pid'],state=s['state'],
  trainer_cmdline=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
  update=status.get('update'),status_time=status.get('time')))
best=max(e['successes'] for e in r['evaluations']);first=next(e for e in r['evaluations'] if e['successes']==best)
r['best_successes']=best;r['first_best_update']=first['update']
r['completed_updates_since_best']=r['latest_commit']['epoch']-first['update']
r['evaluated_update_span_since_best']=r['evaluations'][-1]['update']-first['update']
last=r['evaluations'][-1]
directory=B/last['run']/'evaluations'/f'step_{last["update"]:06d}'
traces=[read(p) for p in directory.glob('*/trace.json')]
assert len(traces)==80
actions=[s for t in traces for s in t['steps']]
collisions=sum(bool(s['collision']) for s in actions)
moving=sum(s['action']!=0 for s in actions)
r['latest_development_collision']=dict(update=last['update'],collisions=collisions,all_actions=len(actions),
 moving_actions=moving,collision_per_all_actions=collisions/len(actions),collision_per_moving_actions=collisions/moving,
 episodes_with_collision=sum(any(s['collision'] for s in t['steps']) for t in traces),episodes=len(traces),
 success_count=sum(bool(t['metrics']['success']) for t in traces))
records=[json.loads(line) for line in (run/'rank0/updates.jsonl').read_text().splitlines() if line.strip()][-10:]
total_actions=sum(x['global_actions'] for x in records);episodes=sum(x['training']['episodes'] for x in records)
r['recent_training_window']=dict(first_update=records[0]['update'],last_update=records[-1]['update'],episodes=episodes,
 success_rate=sum(x['training']['successes'] for x in records)/episodes,
 collision_action_rate=sum(x['training']['collision_action_rate']*x['global_actions'] for x in records)/total_actions,
 actions=total_actions,mean_update_seconds=sum(x['seconds'] for x in records)/len(records))
r['qualification']='Continuous update numbering across v82/v84/v85. Development80 greedy32; original isolatedfixed80 has not been rerun.'
(B/'ppo_plateau_v88.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
