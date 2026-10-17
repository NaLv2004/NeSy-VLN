import collections,json,random,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010')
path=base/'grounding_causality_v29.json';record=json.loads(path.read_text())
record['category_summary']={}
record['swap_protocols']={}
protocol_rows={'same_phase_next_scene':[],'random_within_task':[],'different_expert_action':[]}
for category in ['chair','table']:
    rows=[r for r in record['examples'] if r['category']==category]
    record['category_summary'][category]={'frames':len(rows),'greedy_action_counts':dict(collections.Counter(r['greedy'] for r in rows)),
        'expert_action_counts':dict(collections.Counter(r['action'] for r in rows)),
        'expert_action_agreement':sum(r['greedy']==r['action'] for r in rows)/len(rows),
        'image_swap_changed_count':sum(r['image_swap_greedy_changed'] for r in rows)}
    permutation=list(range(len(rows)));random.Random(2901).shuffle(permutation)
    for mode in protocol_rows:
        for i,row in enumerate(rows):
            j=(i+3)%len(rows) if mode=='same_phase_next_scene' else permutation[i]
            if i==j:continue
            other=rows[j]
            if mode=='different_expert_action' and row['action']==other['action']:continue
            protocol_rows[mode].append({'codes_changed_fraction':sum(a!=b for a,b in zip(row['codes'],other['codes']))/len(row['codes']),
                'greedy_changed':row['greedy']!=other['greedy'],'expert_changed':row['action']!=other['action']})
for mode,rows in protocol_rows.items():
    record['swap_protocols'][mode]={'pairs':len(rows),**{key:sum(r[key] for r in rows)/len(rows)
        for key in ['codes_changed_fraction','greedy_changed','expert_changed']}}
record['qualification']='Initial programs without perceptual history. Same-phase swaps move between demonstrations; random swaps also cross start/middle/end phases. Greedy response sensitivity is not a navigation score.'
path.write_text(json.dumps(record,indent=2));print(json.dumps(record['category_summary'],indent=2))
print(json.dumps(record['swap_protocols'],indent=2))
maintenance=base/'evidence_archives/active_maintenance.json'
if maintenance.exists():
    data=json.loads(maintenance.read_text());p=Path('/proc')/str(data['pid'])/'cmdline'
    print(json.dumps({'maintenance':data,'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None}))
else:print((base/'evidence_archives/active_maintenance_launcher.log').read_text()[-2500:])
