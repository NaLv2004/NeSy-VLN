import hashlib,json,shutil,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/discrete_v4')
target=root/'source_snapshot_initial'
if target.exists():raise RuntimeError('Initial code snapshot already exists; never overwrite it')
target.mkdir()
files={}
for name in ['nesy_vla','scripts','configs','tests']:
    shutil.copytree(root/name,target/name,ignore=shutil.ignore_patterns('__pycache__'))
    for p in sorted((target/name).rglob('*')):
        if p.is_file():files[str(p.relative_to(target))]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'time':time.time(),'files':files,'tests_passed':32,
    'constraint':'All motor-action probabilities are computed from generated rules and discrete predicate tokens.',
    'reference':'https://arxiv.org/html/2603.25903',
    'reference_scope':'Small recurrent symbolic-history model is inspired by ENAP; this is not a reproduction of its HDBSCAN/L-star extraction or visual residual policy.',
    'runs':['runs/discrete_gru','runs/discrete_vlm'],
    'goal_status':'not achieved; closed-loop fixed80 success must reach at least 70 percent'}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'snapshot':str(target),'files':len(files),'tests_passed':32}))
