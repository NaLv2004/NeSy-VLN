"""Audit immutable completed training rollouts1..10 without polling trainers."""
import collections,json,math,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'full_automaton_rl_v62';target=out/'first10_reward_audit.json'
assert not target.exists()
result={'time':time.time(),'scope':'First160 sampled training episodes perarm;notdevelopment/fixed80. No new inference or process poll.','arms':{}}
for arm in ['joint','action_rules_only']:
 folder=out/arm;assert (folder/'step_000010/COMMITTED').exists()
 trials=[]
 for step in range(1,11):
  paths=sorted((folder/'rollouts'/f'step_{step:06d}').glob('*.json'));assert len(paths)==16
  trials.extend(json.loads(p.read_text()) for p in paths)
 terms=collections.defaultdict(float);actions=collections.Counter();returns={True:[],False:[]};stops=0;collisions=0;total=0
 for trial in trials:
  success=bool(trial['metrics'].get('success'));returns[success].append(sum(s['reward'] for s in trial['steps']))
  stops+=trial['steps'][-1]['action']==0
  for row in trial['steps']:
   assert abs(sum(row['reward_terms'].values())-row['reward'])<1e-4
   actions[row['action']]+=1;collisions+=int(row['collision']);total+=1
   for k,v in row['reward_terms'].items():terms[k]+=v
 result['arms'][arm]={'episodes':len(trials),'successes':len(returns[True]),'explicit_stop_episodes':stops,
  'collision_actions':collisions,'actions':total,'action_histogram':dict(actions),
  'reward_term_means_per_episode':{k:v/len(trials) for k,v in terms.items()},
  'return_by_outcome':{str(k):{'count':len(v),'mean':sum(v)/len(v) if v else None,'minimum':min(v) if v else None,'maximum':max(v) if v else None} for k,v in returns.items()},
  'example_terminal_metrics':trials[0]['metrics']}
target.write_text(json.dumps(result,indent=2))
for arm in result['arms'].values():arm.pop('example_terminal_metrics')
print(json.dumps(result,indent=2))
