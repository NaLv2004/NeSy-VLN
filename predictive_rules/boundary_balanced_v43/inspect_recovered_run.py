import json,shutil,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=ROOT/'run_boundary_balanced_v43'
def read(name):return json.loads((out/name).read_text())
sv=read('supervisor.json');ranks=[read(f'rank{i}.status.json') for i in (0,1)]
pids=[]
for pid in [sv.get('pid'),sv.get('worker_pid')]+[r.get('pid') for r in ranks]:
    if pid:
        path=Path(f'/proc/{pid}/cmdline');pids.append(dict(pid=pid,command=path.read_bytes().replace(b'\0',b' ').decode() if path.exists() else ''))
pilot=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:args=path.read_bytes().decode().split('\0')
    except (OSError,UnicodeError):continue
    if str(ROOT/'src_state_sufficiency_v45/probe_representation.py') in args:
        pilot.append(dict(pid=int(path.parent.name),arguments=args))
dev=[json.loads(line) for line in (out/'development.jsonl').read_text().splitlines() if line.strip()]
archives=[]
for folder in sorted(out.glob('dev_*')):
    recipes=list(folder.glob('*/recipe_*.npz'))
    if recipes:archives.append(dict(name=folder.name,recipes=len(recipes),bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
        completed=(folder/'summary.json').exists()))
record=dict(time=time.time(),supervisor=sv,processes=pids,pilot=pilot,
    ranks=[{k:r.get(k) for k in ('rank','update','state','loss','gradient_norm','seconds')} for r in ranks],
    latest=read('latest.COMMITTED.json'),best=read('best.COMMITTED.json'),
    development=[{k:r.get(k) for k in ('update','successes','collision_action_rate','observation_prediction_accuracy')} for r in dev],
    new_archives=archives,free_bytes=shutil.disk_usage(ROOT).free)
print(json.dumps(record))
