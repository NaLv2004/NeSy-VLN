"""Apply the inspected old-table retention list; keep traces, metrics and images."""
import json,re,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_onpolicy_refresh_v60'
inventory=json.loads((OUT/'table_retention_inventory.json').read_text())
audit=OUT/'table_retention_applied.json';assert not audit.exists()
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
entries=[]
for row in inventory['candidates']:
    run=(ROOT/row['run']).resolve();folder=(run/row['evaluation']).resolve()
    assert run.parent==ROOT and folder.parent==run and re.fullmatch(r'dev_\d{6}',folder.name)
    assert folder.name not in row['keep'] and (folder/'summary.json').exists()
    assert not any(str(run) in line and ('train.py' in line or 'supervisor.py' in line) for line in processes)
    files=list(folder.glob('*/program_*.npz'))
    assert len(files)==row['files'] and sum(p.stat().st_size for p in files)==row['bytes']
    for p in files:
        assert not p.is_symlink() and p.resolve().is_relative_to(folder) and (p.parent/'trace.json').exists()
        entries.append(dict(path=str(p),bytes=p.stat().st_size))
record=dict(time=time.time(),entries=entries,free_before=shutil.disk_usage(ROOT).free,
    reason='Preserve latest two and best full rule tables, all traces/metrics/images/checkpoints; remove inspected older generated probability arrays only.')
audit.write_text(json.dumps(record,indent=2))
for e in entries:Path(e['path']).unlink()
record.update(complete=True,free_after=shutil.disk_usage(ROOT).free)
audit.write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='entries'}))
