import hashlib,json,os,shutil,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
src=ROOT/'src_goal_aware_symbols_v52';out=ROOT/'diagnostic_goal_aware_symbols_v52';out.mkdir(exist_ok=True)
old=ROOT/'run_contextual_symbols_v48';commit=json.loads((old/'best.COMMITTED.json').read_text())
assert commit['update']==500,'Inspect a changed best checkpoint instead of silently switching'
source=old/commit['filename'];assert hashlib.sha256(source.read_bytes()).hexdigest()==commit['sha256']
if not (out/'source.pt').exists():os.link(source,out/'source.pt')
assert hashlib.sha256((out/'source.pt').read_bytes()).hexdigest()==commit['sha256']
(out/'source.COMMITTED.json').write_text(json.dumps(dict(commit,source_path=str(source)),indent=2))
if not (src/'predictive_nesy').exists():
    shutil.copytree(ROOT/'src_contextual_symbols_v48/predictive_nesy',src/'predictive_nesy',ignore=shutil.ignore_patterns('__pycache__'))
# Identical immutable model archives can share inode storage without changing
# any filename, bytes, recipe or checksum. Touch only these completed diagnostics.
before=shutil.disk_usage(ROOT).free;seen={};dedup=[]
for name in ('diagnostic_discounted_rules_v46','diagnostic_stable_probability_v50','diagnostic_path_rollout_v51'):
    for path in sorted((ROOT/name).glob('**/program_model_rank*.pt')):
        assert path.resolve().is_relative_to(ROOT/name)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            prior=seen[digest]
            if path.stat().st_ino!=prior.stat().st_ino:
                temp=path.with_suffix('.dedup.tmp');assert not temp.exists()
                os.link(prior,temp);temp.replace(path)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
                dedup.append(dict(path=str(path),shares_with=str(prior),sha256=digest))
        else:seen[digest]=path
report=dict(time=time.time(),source_commit=commit,deduplicated=dedup,free_before=before,free_after=shutil.disk_usage(ROOT).free)
(out/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
