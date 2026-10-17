"""Inventory, then optionally remove obsolete unreferenced intermediate weights."""
import argparse,hashlib,json,re,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_onpolicy_refresh_v60'
parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
texts=[]
for glob in ('run_*/manifest.json','diagnostic_*/*COMMITTED.json','onpolicy_*/*COMMITTED.json','src_*/*.py'):
    for p in ROOT.glob(glob):
        if p.stat().st_size<2_000_000:texts.append((str(p),p.read_text(errors='replace')))
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
rows=[];skips=[]
for folder in ROOT.glob('run_*'):
    if not (folder/'best.COMMITTED.json').exists():continue
    points=sorted(folder.glob('checkpoint_*.pt'))
    if len(points)<2:continue
    # No modification to a run with any remaining live worker.
    if any(str(folder) in line and ('train.py' in line or 'supervisor.py' in line) for line in processes):continue
    statuses=[]
    for n in ('supervisor.json','status.json'):
        p=folder/n
        if p.exists():statuses.append(json.loads(p.read_text()))
    terminal=any(s.get('state') in ('finished','paused','training_budget_complete','complete','stopped') for s in statuses)
    if not terminal:continue
    keep={points[-1].name}
    for p in folder.glob('*.COMMITTED.json'):
        if p.name.startswith('checkpoint_'):continue
        c=json.loads(p.read_text())
        for k in ('filename','path'):
            if isinstance(c.get(k),str):keep.add(Path(c[k]).name)
    # Some older runs record best validation in the manifest/status instead.
    for n in ('best.json','best_development.json'):
        if (folder/n).exists():
            skips.append(dict(run=folder.name,reason='legacy best pointer: retain entire run'));keep.update(p.name for p in points)
    for p in points:
        if p.name in keep or p.is_symlink() or p.stat().st_nlink>1:continue
        commit=p.with_suffix('.COMMITTED.json')
        if not commit.exists():continue
        c=json.loads(commit.read_text());digest=c.get('sha256')
        if not digest:continue
        refs=[n for n,s in texts if str(p) in s]
        if refs:continue
        rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest,commit=str(commit),keep=sorted(keep)))
record=dict(time=time.time(),files=rows,bytes=sum(r['bytes'] for r in rows),free_before=shutil.disk_usage(ROOT).free,skips=skips)
if args.apply:
    previous=json.loads((OUT/'checkpoint_retention_inventory.json').read_text())
    assert {r['path']:r['sha256'] for r in rows}=={r['path']:r['sha256'] for r in previous['files']}
    audit=OUT/'checkpoint_retention_applied.json';assert not audit.exists()
    for r in rows:
        p=Path(r['path']).resolve();assert p.is_relative_to(ROOT) and p.parent.parent==ROOT
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    audit.write_text(json.dumps(record,indent=2))
    for r in rows:Path(r['path']).unlink();Path(r['commit']).unlink()
    record.update(complete=True,free_after=shutil.disk_usage(ROOT).free)
    audit.write_text(json.dumps(record,indent=2))
else:(OUT/'checkpoint_retention_inventory.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
