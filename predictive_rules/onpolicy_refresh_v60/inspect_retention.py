import json,re,shutil,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
rows=[]
for folder in ROOT.glob('run_*'):
    svpath=folder/'supervisor.json'
    if not svpath.exists():continue
    sv=json.loads(svpath.read_text());pid=sv.get('pid');proc=Path(f'/proc/{pid}/cmdline')
    if sv.get('state') not in ('finished','paused') or (proc.exists() and proc.read_bytes()):continue
    evals=sorted(p.parent for p in folder.glob('dev_*/summary.json') if re.fullmatch(r'dev_\d{6}',p.parent.name))
    if not evals:continue
    keep=set(evals[-2:])
    if (folder/'best.COMMITTED.json').exists():keep.add(folder/f'dev_{int(json.loads((folder/"best.COMMITTED.json").read_text())["update"]):06d}')
    for e in evals:
        if e in keep:continue
        files=list(e.glob('*/program_*.npz'))
        if files:rows.append(dict(run=folder.name,evaluation=e.name,files=len(files),bytes=sum(p.stat().st_size for p in files),keep=[p.name for p in keep],terminal_supervisor=sv))
out=ROOT/'diagnostic_onpolicy_refresh_v60'
record=dict(time=time.time(),free=shutil.disk_usage(ROOT).free,candidates=rows,total_bytes=sum(r['bytes'] for r in rows))
(out/'table_retention_inventory.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
