"""Read-only check that category-level goals match evaluator target instances."""
import collections,hashlib,json,time
from pathlib import Path
R=Path('/home/user/xjy/strips_wm_navigation_20261009');B=Path('/home/user/xjy/nesy_vla_20261010')
cfg=json.loads((R/'configs/experiment.json').read_text());lock=json.loads((R/'research/split_lock.json').read_text())
out=B/'goal_observability_v95';out.mkdir(exist_ok=True)
reports={}
for filename in ['train.json','fixed80.json']:
 path=Path(cfg['source_data'])/filename
 assert hashlib.sha256(path.read_bytes()).hexdigest()==lock['hashes'][filename]
 episodes=json.loads(path.read_text());worlds={};rows=[];counts=collections.Counter();category_extra={}
 for e in episodes:
  if e['scene'] not in worlds:
   worlds[e['scene']]=json.loads((Path(cfg['source_world_root'])/e['scene']/'world.json').read_text())
  objects=worlds[e['scene']]['objects'];actual=[o for o in objects if o['name'] in e['goal_object_names']]
  same_category={o['name'] for o in objects if o.get('category')==e['category']}
  excluded=same_category-set(e['goal_object_names'])
  if excluded:category_extra[e['scene']+'|'+e['category']]=sorted(excluded)
  assert len(actual)==len(e['goal_object_names']),e['episode_id']
  # Inspect metadata without inferring categories from object names.
  counts.update({'episodes':1,'targets_total':len(actual),'single_instance_goal':len(actual)==1})
  keys=collections.Counter(k for o in actual for k in o)
  rows.append({'episode_id':e['episode_id'],'scene':e['scene'],'category':e['category'],
    'goal_object_names':e['goal_object_names'],'object_metadata':actual})
 patterns=collections.defaultdict(set)
 for r in rows:patterns[r['scene'],r['category']].add(tuple(sorted(r['goal_object_names'])))
 reports[filename]={'counts':dict(counts),'same_category_instances_excluded_from_success':category_extra,
  'object_categories_by_scene':{scene:dict(collections.Counter(o.get('category','UNLABELED') for o in w['objects'])) for scene,w in worlds.items()},'scene_category_differing_target_sets':[
  {'scene':scene,'category':cat,'target_sets':[list(x) for x in sets]} for (scene,cat),sets in patterns.items() if len(sets)>1],
  'first_examples':rows[:2],'world_keys':{scene:list(w) for scene,w in list(worlds.items())[:1]},
  'target_metadata_keys':list(keys)}
result={'time':time.time(),'scope':'Read-only environment specification audit; no navigation evaluation or input changes. No episode-specific fixed80 metrics inspected.','reports':reports}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
