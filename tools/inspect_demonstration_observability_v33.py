import collections,json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');ref=Path('/home/user/xjy/strips_wm_navigation_20261009')
cfg=json.loads((ref/'configs/experiment.json').read_text());lock=json.loads((ref/'research/split_lock.json').read_text())
allowed=set(lock['train_scenes'])-set(lock['development_scenes'])
tasks=[r for r in json.loads((Path(cfg['source_data'])/'train.json').read_text()) if r['scene'] in allowed]
demo_rows=json.loads((base/'data/lossless_demonstrations_v7/examples.json').read_text())
demo_ids={r['episode_id'] for r in demo_rows};selected=[r for r in tasks if r['episode_id'] in demo_ids]
counts={key:dict(collections.Counter(str(r.get(key)) for r in selected)) for key in selected[0]
    if any(term in key for term in ['visible','visibility','difficulty','distance_group'])}
code=Path(cfg['source_sim_code'])/'large_lab_sim.py';lines=code.read_text().splitlines()
hits=[i for i,s in enumerate(lines) if any(x in s for x in ['def reset(','def distance(','target_render_ids','target_ids','goal_ids'])]
fragments={str(i+1):'\n'.join(lines[max(0,i-2):i+14]) for i in hits[:18]}
result={'time':time.time(),'train_only':True,'training_tasks':len(tasks),'demonstration_episodes':len(selected),
    'episode_keys':list(selected[0]),'examples':selected[:2],'available_visibility_counts':counts,
    'simulator_path':str(code),'simulator_fragments':fragments,'simulator_wrapper':lines[:60] if not hits else None,
    'visibility_summary':dict(collections.Counter('unknown' if r.get('initial_visible_pixels') is None else
        'visible' if r['initial_visible_pixels']>=16 else 'not_visible' for r in selected))}
(base/'demonstration_observability_v33.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['examples','available_visibility_counts']},indent=2))
