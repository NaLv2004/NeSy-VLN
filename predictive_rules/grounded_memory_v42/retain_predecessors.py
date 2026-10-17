"""Bound obsolete generated tables of stopped, explicitly named predecessors.

Retain every checkpoint, trace, image, split, and summary. Best evaluation keeps
its complete table files. Other table archives are regenerable from retained
checkpoints only where those checkpoints exist; records state the limitation.
"""
import json
from pathlib import Path
import re
import shutil
import time

root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015').resolve()
result=[]
for name in ('run_probabilistic_rules_v40','run_stable_onpolicy_v41'):
    out=(root/name).resolve();assert out.parent==root
    supervisor=json.loads((out/'supervisor.json').read_text())
    p=Path(f'/proc/{supervisor["pid"]}/cmdline')
    assert not p.exists() or not p.read_bytes(), f'{name} still running'
    best=json.loads((out/'best.COMMITTED.json').read_text())
    keep=out/f'dev_{best["update"]:06d}'
    for summary in sorted(out.glob('dev_*/summary.json')):
        folder=summary.parent
        assert re.fullmatch(r'dev_\d{6}',folder.name)
        if folder==keep:continue
        files=list(folder.glob('*/program_*.npz'));entries=[]
        for file in files:
            assert file.resolve().is_relative_to(out) and not file.is_symlink()
            assert re.fullmatch(r'program_\d{2}\.npz',file.name)
            assert (file.parent/'trace.json').is_file()
            entries.append(dict(path=str(file.relative_to(folder)),bytes=file.stat().st_size))
        if not entries:continue
        record=dict(time=time.time(),reason='Stopped predecessor retains complete best-evaluation tables. All checkpoints, splits, traces, metrics, images and audits preserved.',
            full_probability_tables_replayable=False,files=entries,total_bytes=sum(e['bytes'] for e in entries))
        (folder/'program_table_retention.json').write_text(json.dumps(record,indent=2))
        for file in files:file.unlink()
        result.append(dict(run=name,evaluation=folder.name,bytes=record['total_bytes']))
    with (out/'storage_retention.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(),removed=[r for r in result if r['run']==name]))+'\n')
print(json.dumps(dict(removed=result,free_gib=shutil.disk_usage(root).free/2**30)))
