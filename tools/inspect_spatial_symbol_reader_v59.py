import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/spatial_symbol_reader_v59')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json');meta=read(out/'launcher.json')
if old and old['launcher']['pid']==meta['pid'] and time.time()-old['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
proc=Path(f'/proc/{meta["pid"]}/cmdline');result=read(out/'result.json') or read(out/'progress.json') or {}
record={'time':time.time(),'launcher':meta,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
 'status':read(out/'status.json'),'summary':{k:{'parameters':v['parameters'],'measurements':[{field:(value if field not in ['fit','audit'] else {x:y for x,y in value.items() if x!='predictions'}) for field,value in row.items()} for row in v['measurements']]} for k,v in result.get('arms',{}).items()},
 'tail':(out/'driver.log').read_text(errors='replace')[-1000:]}
(out/'inspection.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
