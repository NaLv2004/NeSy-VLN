"""Authorized bounded pruning: retain best, final, latest two and all references."""
import hashlib,json,os,re,shutil,sys,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
NAMES=('run_raw_grounding_v31','run_rule_planning_v32','run_action_grounded_v33')
apply='--apply' in sys.argv
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def processes():
    hits=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) in (os.getpid(),os.getppid()):continue
        try:cmd=p.read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeError):continue
        if any(str(ROOT/name) in cmd for name in NAMES):hits.append(dict(pid=p.parent.name,cmd=cmd))
    return hits
assert not processes(),'Old run still has a live owner'
candidates=[];kept={}
for name in NAMES:
    folder=ROOT/name;points=sorted(folder.glob('checkpoint_*.pt'));assert points
    keep={p.name:'latest_two' for p in points[-2:]}
    summaries=[]
    for p in folder.glob('development_step_*/summary.json'):
        row=json.loads(p.read_text())
        if row.get('episodes')==80:summaries.append((row['successes'],int(p.parent.name.rsplit('_',1)[1])))
    assert summaries
    best=max(n for n,_ in summaries)
    for n,u in summaries:
        if n==best:keep[f'checkpoint_{u:06d}.pt']='best_full80'
    for p in points:
        if p.name in keep:continue
        assert p.resolve().parent==folder.resolve() and not p.is_symlink()
        candidates.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,run=name,references=[]))
    kept[name]=keep
# A conservative text-reference audit covers source and lineage manifests.
# Traces retain their own archived programs and do not depend on optimizer files.
for base,dirs,files in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d!='__pycache__' and not d.startswith(('development_step_','dev_','counterfactual_'))]
    for file in files:
        path=Path(base)/file
        if path.suffix not in ('.json','.py','.md','.jsonl'):continue
        if path.name in ('trace.json','cache.json','rank0.jsonl','rank1.jsonl'):continue
        if path.stat().st_size>10<<20:continue
        try:content=path.read_text()
        except (UnicodeError,OSError):continue
        for item in candidates:
            candidate=Path(item['path'])
            # Own checkpoint commit, own evaluation journal, and this retention
            # report are inventories rather than dependent implementations.
            if path.parent==candidate.parent or path.name.startswith('retention_v53') or path.name.endswith('_monitor_latest.json'):continue
            if item['path'] in content or item['sha256'] in content or (item['run'] in content and candidate.name in content):
                item['references'].append(str(path))
deletable=[v for v in candidates if not v['references']]
report=dict(time=time.time(),apply=apply,free_before=shutil.disk_usage(ROOT).free,retained=kept,
    candidates=candidates,remove=deletable,estimated_bytes=sum(v['bytes'] for v in deletable),
    rule='Remove only unreferenced intermediate weights in three verified terminal runs; retain best/full latest/source references and every trace/result.')
destination=ROOT/'retention_v53.json';destination.write_text(json.dumps(report,indent=2))
if apply:
    assert not processes()
    for item in deletable:
        path=Path(item['path']);assert path.parent.name in NAMES and path.resolve().parent==ROOT/path.parent.name
        assert sha(path)==item['sha256']
        path.unlink()
    report.update(free_after=shutil.disk_usage(ROOT).free,removed_count=len(deletable))
    destination.write_text(json.dumps(report,indent=2))
print(json.dumps(report))
