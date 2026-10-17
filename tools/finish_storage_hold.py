import json,os,signal,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');record=json.loads((base/'storage_hold_20261011.json').read_text())
known={r['pid'] for r in record['processes']}
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:pid=int(path.parent.name);cmd=path.read_bytes().replace(b'\0',b' ').decode()
    except (OSError,ValueError):continue
    if pid==os.getpid() or pid in known or 'scripts/train.py --config' not in cmd or str(base) not in cmd:continue
    if not any(name in cmd for name in ['symbolic_joint_v9','symbolic_robust_v7','symbolic_vision_v8']):continue
    group=os.getpgid(pid);assert group==pid
    os.killpg(group,signal.SIGSTOP)
    record['processes'].append({'pid':pid,'pgid':group,'cmd':cmd,'supervisor':False,'worker':True})
opened=set()
for fd in Path('/proc').glob('[0-9]*/fd/*'):
    try:opened.add(fd.resolve())
    except OSError:pass
removed=[]
for relative in ['symbolic_robust_v7/runs/robust_multieffect/latest','symbolic_joint_v9/runs/joint/latest']:
    folder=base/relative;marker=json.loads((folder/'COMMITTED').read_text())
    for path in folder.glob('checkpoint.*.tmp'):
        assert path.resolve().is_relative_to(base) and path.name!=marker['checkpoint_file']
        if path.resolve() in opened:continue
        removed.append({'path':str(path),'bytes':path.stat().st_size,'reason':'Failed uncommitted temporary file, not open in any process'})
        path.unlink()
record.update(extra_workers_held=True,failed_uncommitted_temporary_files_removed=removed)
(base/'storage_hold_20261011.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
