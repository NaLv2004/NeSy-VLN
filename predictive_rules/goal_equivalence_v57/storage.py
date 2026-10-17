"""Bound large generated evaluation tables; retain all traces and summaries."""
import json
from pathlib import Path
import re
import time

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')


def retain_evaluation_tables(out):
    out=Path(out).resolve()
    assert out.parent==ROOT and out.name=='run_goal_equivalence_v57'
    complete=sorted(p.parent for p in out.glob('dev_*/summary.json') if re.fullmatch(r'dev_\d{6}',p.parent.name))
    keep=set(complete[-2:])
    best=out/'best.COMMITTED.json'
    if best.exists():keep.add(out/f'dev_{int(json.loads(best.read_text())["update"]):06d}')
    removed=[]
    for folder in complete:
        if folder in keep:continue
        files=list(folder.glob('*/program_*.npz'))
        if not files:continue
        entries=[]
        for file in files:
            assert file.resolve().is_relative_to(out) and not file.is_symlink()
            assert re.fullmatch(r'program_\d{2}\.npz',file.name)
            assert (file.parent/'trace.json').is_file()
            entries.append(dict(path=str(file.relative_to(folder)),bytes=file.stat().st_size))
        record=dict(time=time.time(),reason='Retain complete tables for latest two evaluations and current best; retain every historical trace, proof witness, metric, image, and audit.',
                    full_probability_tables_replayable=False,files=entries,total_bytes=sum(e['bytes'] for e in entries))
        temp=folder/'program_table_retention.tmp'
        temp.write_text(json.dumps(record,indent=2));temp.replace(folder/'program_table_retention.json')
        for file in files:file.unlink()
        removed.append(dict(evaluation=folder.name,bytes=record['total_bytes'],files=len(entries)))
    if removed:
        with (out/'storage_retention.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),removed=removed))+'\n')
    return removed


if __name__=='__main__':
    print(json.dumps(retain_evaluation_tables(ROOT/'run_goal_equivalence_v57')))
