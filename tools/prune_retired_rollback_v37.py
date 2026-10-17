"""Remove only old atomic-save rollback files; all committed models stay intact."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010').resolve();records=[]
audit=base/'retired_rollback_pruning_v37.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def live(run):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:
            command=p.read_bytes()
            if str(run).encode() in command and (b'scripts/train.py' in command or b'scripts/supervise.py' in command):return True
        except OSError:pass
    return False

protected=set()
for marker in base.rglob('COMMITTED'):
    m=json.loads(marker.read_text())
    if 'checkpoint_file' in m:protected.add((marker.parent/m['checkpoint_file']).resolve())
for family in ['symbolic_teacher_v25','symbolic_task_v15','symbolic_branch_v17','symbolic_conjunctive_v23']:
    for run in (base/family/'runs').glob('*'):
        if not (run/'status.json').exists() or json.loads((run/'status.json').read_text()).get('state')!='complete' or live(run):continue
        for name in ['latest','best_development']:
            marker=run/name/'COMMITTED'
            if not marker.exists():continue
            m=json.loads(marker.read_text());keep=marker.parent/m['checkpoint_file']
            assert sha(keep)==m['checkpoint_sha256']
            for p in marker.parent.glob('checkpoint.e*.pt'):
                if p.resolve() in protected or p.is_symlink() or p.stat().st_nlink>1:continue
                assert p.resolve().is_relative_to(run) and p.parent==marker.parent
                records.append({'run':str(run),'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,
                    'retained_file':str(keep),'retained_sha256':m['checkpoint_sha256']})
if '--apply' not in sys.argv:
    assert not audit.exists(),'Existing cleanup audit must be inspected'
    audit.write_text(json.dumps({'state':'planned','records':records,'time':time.time()},indent=2))
    print(json.dumps({'state':'planned','records':records,'gib':sum(r['bytes'] for r in records)/2**30},indent=2))
else:
    plan=json.loads(audit.read_text());assert plan['state']=='planned' and plan['records']==records
    for row in records:
        assert not live(Path(row['run'])) and sha(Path(row['retained_file']))==row['retained_sha256']
    plan.update(state='verified_removing');audit.write_text(json.dumps(plan,indent=2))
    for row in records:
        p=Path(row['path']);assert p.resolve().is_relative_to(base) and p.resolve() not in protected
        p.unlink()
    plan.update(state='complete',finished=time.time(),released_gib=sum(r['bytes'] for r in records)/2**30,free_gib=shutil.disk_usage(base).free/2**30)
    audit.write_text(json.dumps(plan,indent=2));print(json.dumps({k:v for k,v in plan.items() if k!='records'},indent=2))
