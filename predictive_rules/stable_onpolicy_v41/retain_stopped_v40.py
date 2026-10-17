"""Retain best and final evaluation tables of the safely stopped predecessor."""
import json
from pathlib import Path
import re
import shutil
import time

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=(root/'run_probabilistic_rules_v40').resolve()
status=json.loads((out/'supervisor.json').read_text())
p=Path(f'/proc/{status["pid"]}/cmdline')
assert not p.exists() or not p.read_bytes(),'Predecessor still running'
complete=sorted(p.parent for p in out.glob('dev_*/summary.json') if re.fullmatch(r'dev_\d{6}',p.parent.name))
best=json.loads((out/'best.COMMITTED.json').read_text())
keep={complete[-1],out/f'dev_{best["update"]:06d}'}
removed=[]
for folder in complete:
    if folder in keep:continue
    files=list(folder.glob('*/program_*.npz'))
    if not files:continue
    entries=[]
    for file in files:
        assert file.resolve().is_relative_to(out) and not file.is_symlink()
        assert re.fullmatch(r'program_\d{2}\.npz',file.name) and (file.parent/'trace.json').is_file()
        entries.append(dict(path=str(file.relative_to(folder)),bytes=file.stat().st_size))
    record=dict(time=time.time(),reason='Stopped predecessor: retain full best and final evaluation tables; preserve every trace, metric, image, audit and retained checkpoint.',
        full_probability_tables_replayable=False,files=entries,total_bytes=sum(e['bytes'] for e in entries))
    (folder/'program_table_retention.json').write_text(json.dumps(record,indent=2))
    for file in files:file.unlink()
    removed.append(dict(evaluation=folder.name,bytes=record['total_bytes']))
result=dict(time=time.time(),removed=removed,free_gib=shutil.disk_usage(root).free/2**30)
with (out/'storage_retention.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
print(json.dumps(result))
