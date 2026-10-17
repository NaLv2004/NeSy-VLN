import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/program_causality_v60');read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json');meta=read(out/'launcher.json')
if old and old['launcher']['pid']==meta['pid'] and time.time()-old['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
proc=Path(f'/proc/{meta["pid"]}/cmdline');result=read(out/'result.json') or {}
r={'time':time.time(),'launcher':meta,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
 'status':read(out/'status.json'),'arms':result.get('arms',{}),'tail':(out/'driver.log').read_text(errors='replace')[-1200:]}
(out/'inspection.json').write_text(json.dumps(r,indent=2))
for arm in r['arms'].values():
 arm['development_summary']={k:v for k,v in arm['development_summary'].items() if k!='groups'}
print(json.dumps(r,indent=2))
