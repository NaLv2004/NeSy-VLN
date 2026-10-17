import json,time,shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/calibrated_vision_v10');saved=root/'current_status.json'
def read(p):return json.loads(p.read_text()) if p.exists() else None
previous=read(saved)
if previous and previous.get('stable') and time.time()-previous['time']<600:
    print(json.dumps({'skipped':'stable minimum600 seconds','last_time':previous['time']}));raise SystemExit
result={'time':time.time(),'jobs':{},'free_gib':shutil.disk_usage(root).free/2**30}
for record in read(root/'pilot_processes.json'):
    out=Path(record['out']);cmd=Path(f"/proc/{record['pid']}/cmdline")
    item={'process':record,'live':cmd.read_bytes().replace(b'\0',b' ').decode() if cmd.exists() else None,
        'status':read(out/'status.json'),'checkpoint':read(out/'latest/COMMITTED')}
    log=out/'rank0.jsonl'
    if log.exists():
        row=json.loads(log.read_text().splitlines()[-1]);item['last_update']={k:row[k] for k in ['update','seconds','losses','gpu_peak_gib','gradient_by_module']}
    item['tail']=(out/'pilot.log').read_text(errors='replace')[-1200:]
    item['evaluations']={step.name:{s:read(step/s/'summary.json') for s in ['development','fixed80']} for step in (out/'evaluations').glob('step_*')}
    result['jobs'][record['name']]=item
result['stable']=all(j['live'] and (j['status'] or {}).get('update',0)>0 for j in result['jobs'].values())
saved.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
