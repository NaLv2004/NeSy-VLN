import json,collections,statistics
from pathlib import Path
root=Path('/home/user/xjy/strips_wm_navigation_20261009')
rows=json.loads((root/'data/search_cost_supervision_v2/examples.json').read_text())
lock=json.loads((root/'research/split_lock.json').read_text())
env=json.loads((root/'configs/experiment.json').read_text())
tasks={e['episode_id']:e for e in json.loads((Path(env['source_data'])/'train.json').read_text())}
episodes=collections.defaultdict(list)
for row in rows:episodes[row['episode_id']].append(row)
groups=collections.defaultdict(list);actions=collections.Counter();files=[]
for key,sequence in episodes.items():
    sequence.sort(key=lambda r:r['image']);task=tasks[key]
    assert task['scene'] not in lock['development_scenes'] and task['scene'] not in lock['fixed_scenes']
    assert sequence[-1]['action']==0
    labels=[r['action'] for r in sequence];actions.update(labels)
    groups[(task['domain'],task['distance_group'])].append(sequence)
    if len(files)<2:files.append({'episode':key,'images':[r['image'] for r in sequence],'actions':labels})
result={'episodes':len(episodes),'steps':len(rows),'action_counts':dict(sorted(actions.items())),
    'majority_action_fraction':max(actions.values())/len(rows),'mean_steps':statistics.mean(map(len,episodes.values())),
    'groups':{'/'.join(key):{'episodes':len(value),'steps':sum(map(len,value)),
        'actions':dict(collections.Counter(r['action'] for seq in value for r in seq))} for key,value in groups.items()},
    'example_ordering':files,'all_episodes_isolated_from_validation':True}
out=Path('/home/user/xjy/nesy_vla_20261010/discrete_v4/diagnostics/demonstration_distribution.json')
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
