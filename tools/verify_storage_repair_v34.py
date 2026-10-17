import json,time,shutil
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');read=lambda p:json.loads(p.read_text())
rows=[]
for family,names in [('symbolic_teacher_v25',['ema_control','synchronous']),('symbolic_recovery_v28',['base_control','recovery_mixture'])]:
    for name in names:
        run=base/family/'runs'/name;supervisor=read(run/'supervisor.json');pid=supervisor.get('pid')
        proc=Path(f'/proc/{pid}/cmdline') if pid else None;status=read(run/'status.json')
        rows.append({'run':str(run),'status':status,'commit':read(run/'latest/COMMITTED'),
            'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc is not None and proc.exists() else None,
            'storage_waits':{str(p.relative_to(run)):read(p) for p in run.rglob('WAITING_FOR_STORAGE.json')}})
report={'time':time.time(),'reason':'Verify storage-blocked live processes advanced after validated checkpoint/evidence cleanup',
    'free_gib':shutil.disk_usage(base).free/2**30,'jobs':rows}
(base/'storage_repair_v34.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
