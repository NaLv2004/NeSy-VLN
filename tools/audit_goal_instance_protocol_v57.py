"""Read only: check whether category-only policy observations identify goals."""
import json,re,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010')
c=json.loads((b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json').read_text())
experiment=json.loads((Path(c['reference_root'])/'configs/experiment.json').read_text())
episodes=json.loads((Path(experiment['source_data'])/'train.json').read_text())
worlds={}
for ep in episodes:
 if ep['scene'] not in worlds:
  worlds[ep['scene']]=json.loads((Path(experiment['source_world_root'])/ep['scene']/'world.json').read_text())
checks=[]
for ep in episodes:
 objects=worlds[ep['scene']]['objects']
 category=ep['category'];goals=ep.get('goal_object_names',[])
 candidates=[o['name'] for o in objects if o.get('category')==category]
 checks.append({'episode_id':ep['episode_id'],'scene':ep['scene'],'category':category,
 'goals':goals,'all_category_objects':candidates,'match':set(goals)==set(candidates)})
result={'time':time.time(),'episodes':len(checks),'all_category_goal_matches':sum(x['match'] for x in checks),
 'mismatch_examples':[x for x in checks if not x['match']][:5],
 'episode_example':episodes[0],'object_examples':worlds[episodes[0]['scene']]['objects'][:3]}
(b/'goal_instance_protocol_v57.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
for file in [Path('/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1/objectnav_rl/simulator.py')]:
 if not file.exists():continue
 lines=file.read_text().splitlines();selected=set()
 for i,line in enumerate(lines):
  if re.search('target_render_ids|goal_object|def distance|def reset|viewpoints|target_names',line):selected.update(range(max(0,i-3),min(len(lines),i+12)))
 print(json.dumps({'file':str(file),'lines':[f'{i+1}:{lines[i]}' for i in sorted(selected)]},indent=2))
