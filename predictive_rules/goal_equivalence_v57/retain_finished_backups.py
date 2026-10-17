"""Remove only inspected, unevaluated penultimate backups of four finished runs."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
expected={
 'run_hybrid_replay_v54/checkpoint_001900.pt':'14a3113c7266c3c0c3fe5fdf7cdec6fd2055741831aa8aeee504582b6227d206',
 'run_temporal_symbols_v55/checkpoint_001900.pt':'2abbc193b250cd983c180f1de51b4b8575601172cfc66291d8523f5a32bfbbe8',
 'run_goal_aware_symbols_v52/checkpoint_002900.pt':'e6c9d7a08f293e66466e2f1332e06a6640626e990d7e7e60b1831cd8217fb2d6',
 'run_contextual_symbols_v48/checkpoint_002900.pt':'f740b18dc31acdac60bee371fee793ca2f9b0f117337175bfd53ba515327f874'}
references=list(ROOT.glob('run_*/manifest.json'))+list(ROOT.glob('diagnostic_*/source*.json'))+list(ROOT.glob('src_*/*.py'))
texts={p:p.read_text(errors='replace') for p in references if p.stat().st_size<2_000_000}
entries=[]
for name,digest in expected.items():
    file=(ROOT/name).resolve();assert file.is_relative_to(ROOT) and file.parent.parent==ROOT and not file.is_symlink()
    parent=file.parent;sv=json.loads((parent/'supervisor.json').read_text())
    process=Path(f'/proc/{sv["pid"]}/cmdline')
    assert sv['state']=='finished' and (not process.exists() or not process.read_bytes())
    keep={json.loads((parent/m).read_text())['filename'] for m in ('best.COMMITTED.json','latest.COMMITTED.json')}
    assert file.name not in keep and not (parent/('dev_'+file.stem.split('_')[1])).exists()
    commit=json.loads(file.with_suffix('.COMMITTED.json').read_text())
    assert commit['sha256']==digest and file.stat().st_nlink==1
    assert hashlib.sha256(file.read_bytes()).hexdigest()==digest
    hits=[str(p) for p,s in texts.items() if str(file) in s]
    assert not hits,dict(checkpoint=str(file),references=hits)
    entries.append(dict(path=str(file),bytes=file.stat().st_size,commit=commit))
audit=ROOT/'diagnostic_goal_equivalence_v57/finished_backup_retention.json'
assert not audit.exists(),'Do not repeat completed cleanup'
result=dict(time=time.time(),reason='Finished runs: preserve best and final; remove only unevaluated, unreferenced penultimate rolling backups',
    protected_live_runs=['run_definition_rules_v56','run_goal_equivalence_v57'],
    entries=entries,bytes=sum(e['bytes'] for e in entries),free_before=shutil.disk_usage(ROOT).free)
audit.write_text(json.dumps(result,indent=2))
for entry in entries:
    p=Path(entry['path']);p.unlink();p.with_suffix('.COMMITTED.json').unlink()
result.update(free_after=shutil.disk_usage(ROOT).free,complete=True)
audit.write_text(json.dumps(result,indent=2));print(json.dumps(result))
