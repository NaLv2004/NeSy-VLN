import json,time,statistics
from pathlib import Path

base=Path('/home/user/xjy/nesy_vla_20261010');study=base/'branch_temperature_v18'
records=json.loads((study/'processes.json').read_text()) if (study/'processes.json').exists() else []
result={'time':time.time(),'processes':{},'observability':{}}
for record in records:
    out=study/record['name'];proc=Path('/proc')/str(record['pid'])/'cmdline'
    live=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
    value=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
    rows=[json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()] if (out/'progress.jsonl').exists() else []
    result['processes'][record['name']]={'pid':record['pid'],'live':live,'result':value,
        'last_progress':rows[-1] if rows else None,'tail':(out/'probe.log').read_text()[-1600:]}
for name,folder in [('full',base/'data/paired_goals_v12'),('probe',base/'paired_goal_probe_v12/input_snapshot')]:
    pairs=json.loads((folder/'pairs.json').read_text());branches=[b for p in pairs for b in p['branches']]
    result['observability'][name]={'pairs':len(pairs),'branches':len(branches),
        'initial_target_visible_at_least_16_pixels':sum(b['initial_metrics']['visible_pixels']>=16 for b in branches),
        'initial_target_not_visible':sum(b['initial_metrics']['visible_pixels']==0 for b in branches),
        'both_goals_visible':sum(all(b['initial_metrics']['visible_pixels']>=16 for b in p['branches']) for p in pairs),
        'both_goals_invisible':sum(all(b['initial_metrics']['visible_pixels']==0 for b in p['branches']) for p in pairs),
        'example_initial_metrics':[b['initial_metrics'] for b in branches[:6]]}
(base/'temperature_and_observability_inspection.json').write_text(json.dumps(result,indent=2))
for value in result['processes'].values():
    r=value['result']
    if r:
        value['result']={k:r[k] for k in ['state','initial','final','route_temperature'] if k in r}
print(json.dumps(result,indent=2))
