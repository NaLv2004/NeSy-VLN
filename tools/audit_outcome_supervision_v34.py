import json,hashlib,collections,shutil,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');ref=Path('/home/user/xjy/strips_wm_navigation_20261009')
read=lambda p:json.loads(p.read_text())
probe=read(b/'visibility_probe_breakdown_v33.json')
print(json.dumps({'visibility_probe':{k:v['summary'] for k,v in probe['results'].items()}},indent=2))
p=ref/'data/search_cost_supervision_v3/execution_outcomes.json';rows=read(p)
manifest=read(p.parent/'manifest.json');counts=collections.Counter()
for r in rows:
    counts.update(r.keys())
selected=read(b/'conjunction_learning_v31/dataset.json')['examples']
ids={r['episode_id'] for r in rows}
report={'time':time.time(),'manifest':manifest,'rows':len(rows),'keys':dict(counts),
 'first':rows[0], 'last':rows[-1], 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'selected_probe_episodes_with_outcomes':sum(r['id'] in ids for r in selected),
 'maintenance':read(b/'evidence_archives/active_maintenance.json'),'free_gib':shutil.disk_usage(b).free/2**30}
(b/'outcome_supervision_audit_v34.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'outcome_audit':report},indent=2))
