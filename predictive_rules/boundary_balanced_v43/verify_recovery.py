import json,shutil,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
run=ROOT/'run_boundary_balanced_v43'
sv=json.loads((run/'supervisor.json').read_text());assert sv['state']=='running'
statuses=[json.loads((run/f'rank{rank}.status.json').read_text()) for rank in (0,1)]
pids=[sv['pid'],sv['worker_pid']]+[r['pid'] for r in statuses]
for pid in pids:
    assert 'src_boundary_balanced_v43/' in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
manifest=json.loads((run/'manifest.json').read_text())
assert manifest['resume_commit']['update']==4500
assert all(r['update']>4500 for r in statuses)
record=dict(time=time.time(),pids=pids,resume=manifest['resume_commit'],
    ranks=[{key:r.get(key) for key in ('rank','update','state','loss','gradient_norm','seconds')} for r in statuses],
    free_bytes=shutil.disk_usage(ROOT).free,
    latest_completed_evaluation=json.loads((run/'development.jsonl').read_text().splitlines()[-1]))
(run/'disk_repair_verified.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
