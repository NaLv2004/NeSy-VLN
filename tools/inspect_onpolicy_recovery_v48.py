import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/data/onpolicy_recovery_v48');read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json')
if old and time.time()-old['time']<600:print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
rows=[]
for j in read(out/'launcher.json'):
 p=Path(f'/proc/{j["pid"]}/cmdline');run=out/f'rank_{j["rank"]}';done=read(run/'COMPLETE')
 rows.append({'rank':j['rank'],'pid':j['pid'],'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,'status':read(run/'status.json'),'complete':{k:v for k,v in done.items() if k!='records'} if done else None,'tail':(run/'driver.log').read_text(errors='replace')[-1700:]})
r={'time':time.time(),'jobs':rows};(out/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
