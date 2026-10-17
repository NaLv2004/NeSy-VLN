"""Verify assets and prerequisites, then drain only the owned v42 run."""
import hashlib,json,subprocess,time,shutil
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
asset=ROOT/'assets/dinov2_small'
record=json.loads((asset/'provenance.json').read_text())
for row in record['files']:
    path=asset/row['name']
    assert path.stat().st_size==row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
print(json.dumps(dict(asset_verified=True,bytes=sum(r['bytes'] for r in record['files']),
    free_gib=shutil.disk_usage(ROOT).free/2**30,time=time.time())),flush=True)
contract=json.loads((ROOT/'src_boundary_balanced_v43/sampling_contracts.json').read_text())
assert contract['passed']
old=ROOT/'run_grounded_memory_v42'
summaries=[json.loads(line) for line in (old/'development.jsonl').read_text().splitlines() if line]
baseline=dict(development=summaries,latest=json.loads((old/'latest.COMMITTED.json').read_text()),
    status=json.loads((old/'rank0.status.json').read_text()),time=time.time())
(ROOT/'src_boundary_balanced_v43/predecessor_before_drain.json').write_text(json.dumps(baseline,indent=2))
print(json.dumps(dict(development=[dict(update=r['update'],successes=r['successes']) for r in summaries],
    before_drain_update=baseline['status']['update'],state=baseline['status']['state'])),flush=True)
subprocess.run(['python3','-u',str(ROOT/'src_grounded_memory_v42/drain_owned.py')],check=True)
