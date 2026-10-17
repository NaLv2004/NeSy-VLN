import json,shutil,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
result=dict(time=time.time(),disk=shutil.disk_usage(ROOT)._asdict(),runs={})
for name in ('run_probabilistic_rules_v40','run_stable_onpolicy_v41','run_grounded_memory_v42','run_boundary_balanced_v43'):
    run=ROOT/name;record={}
    for key in ('supervisor.json','rank0.status.json','latest.COMMITTED.json','best.COMMITTED.json'):
        p=run/key
        if p.exists():
            try:record[key]=json.loads(p.read_text())
            except ValueError:record[key]='incomplete JSON'
    sv=record.get('supervisor.json',{});record['processes']={}
    for pid in [sv.get('pid'),sv.get('worker_pid'),record.get('rank0.status.json',{}).get('pid')]:
        if not pid:continue
        p=Path(f'/proc/{pid}/cmdline')
        record['processes'][pid]=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
    record['table_folders']=[dict(folder=p.name,bytes=sum(f.stat().st_size for f in p.glob('*/program_*.npz')),complete=(p/'summary.json').exists()) for p in sorted(run.glob('dev_*')) if p.is_dir()]
    result['runs'][name]=record
p=ROOT/'run_boundary_balanced_v43/attempt_0.log'
with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));result['log_tail']=f.read().decode(errors='replace')
print(json.dumps(result))
