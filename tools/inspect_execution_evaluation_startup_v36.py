import json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'execution_development_v36'
read=lambda p:json.loads(p.read_text()) if p.exists() else None
saved=read(out/'startup_inspection.json')
if saved and time.time()-saved['time']<600:
    print(json.dumps({'skipped':'minimum600seconds','next_check':saved['time']+600}));raise SystemExit
launcher=read(out/'launcher.json');pid=launcher['pid'];cmd=Path(f'/proc/{pid}/cmdline')
children=read(out/'processes.json');sim=None
if children:
    p=Path(f'/proc/{children["simulator_pid"]}/cmdline');sim=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
result={'time':time.time(),'launcher':launcher,'live':cmd.read_bytes().replace(b'\0',b' ').decode() if cmd.exists() else None,
    'simulator_live':sim,'status':read(out/'status.json'),
    'committed_episodes':sum(1 for p in out.glob('*/*/evaluation.json')),
    'tail':(out/'driver.log').read_text(errors='replace')[-1500:]}
(out/'startup_inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
