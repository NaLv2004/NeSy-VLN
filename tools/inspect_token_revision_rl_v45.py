import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/token_revision_rl_v45')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
previous=read(out/'inspection.json')
repair=read(out.parent/'storage_repair_verification_v46.json')
last=max((previous or {}).get('time',0),(repair or {}).get('time',0))
if last and time.time()-last<600:
    print(json.dumps({'skipped':'minimum600seconds','next_check':last+600}));raise SystemExit
def live(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
meta=read(out/'launcher.json');supervisor=read(out/'supervisor.json');children=read(out/'processes.json')
lines=(out/'updates.jsonl').read_text().splitlines() if (out/'updates.jsonl').exists() else []
record={'time':time.time(),'launcher':meta,'supervisor':supervisor,'supervisor_live':live(meta['pid']),
        'trainer_live':live((supervisor or {}).get('pid')),'status':read(out/'status.json'),
        'latest_update':json.loads(lines[-1]) if lines else None,'commit':read(out/'latest/COMMITTED'),
        'simulators':{str(pid):live(pid) for pid in (children or {}).get('simulator_pids',[])},
        'initial_fidelity':read(out/'initial_fidelity.json'),'tail':(out/'driver.log').read_text(errors='replace')[-2500:] if (out/'driver.log').exists() else ''}
(out/'inspection.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
