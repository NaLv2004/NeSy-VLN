"""Remove only bulky exports of one abandoned incomplete old evaluation."""
import hashlib,json,os,re,shutil,sys,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
RUN=ROOT/'run_boundary_balanced_v43';FOLDER=RUN/'dev_004500';apply='--apply' in sys.argv
assert FOLDER.resolve().parent==RUN and FOLDER.is_dir()
assert not (FOLDER/'summary.json').exists(),'Complete evaluation must be retained by ordinary policy'
status=json.loads((RUN/'status.json').read_text());assert status['state'] in ('finished','paused') and status['update']>4500
best=json.loads((RUN/'best.COMMITTED.json').read_text());assert best['update']!=4500
latest=json.loads((RUN/'latest.COMMITTED.json').read_text());assert latest['update']>4500
for path in Path('/proc').glob('[0-9]*/cmdline'):
    if int(path.parent.name) in (os.getpid(),os.getppid()):continue
    try:cmd=path.read_bytes().decode(errors='replace')
    except OSError:continue
    assert str(RUN) not in cmd and str(ROOT/'src_boundary_balanced_v43') not in cmd,cmd
refs=[]
for base,ds,fs in os.walk(ROOT):
    ds[:]=[d for d in ds if not d.startswith(('dev_','development_','counterfactual_','__pycache__'))]
    for name in fs:
        path=Path(base)/name
        if path.suffix not in ('.py','.json','.md') or path.stat().st_size>5<<20:continue
        if path.name.startswith(('prune_abandoned_exports','abandoned_export_retention')):continue
        try:content=path.read_text()
        except (UnicodeError,OSError):continue
        if str(FOLDER)+'/' in content:refs.append(str(path))
assert not refs,refs
files=sorted(FOLDER.glob('*/program_*.npz'));assert files
entries=[]
for file in files:
    assert file.resolve().is_relative_to(FOLDER) and not file.is_symlink()
    assert re.fullmatch(r'program_\d{2}\.npz',file.name)
    entries.append(dict(path=str(file),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
report=dict(time=time.time(),apply=apply,folder=str(FOLDER),terminal_status=status,best=best,latest=latest,
    incomplete_evaluation=True,files=entries,bytes=sum(r['bytes'] for r in entries),free_before=shutil.disk_usage(ROOT).free,
    preserved='Every checkpoint, complete evaluation, summary, image, trace and proof witness. Only abandoned full numeric table exports are removed.',
    limitation='This abandoned partial evaluation no longer supports replay of its complete probability tables.')
destination=ROOT/'abandoned_export_retention_v55.json';destination.write_text(json.dumps(report,indent=2))
if apply:
    for item in entries:
        path=Path(item['path']);assert path.resolve().is_relative_to(FOLDER)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];path.unlink()
    report['free_after']=shutil.disk_usage(ROOT).free;destination.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='files'}))
