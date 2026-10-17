import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/visual_vocabulary_probe_v53');read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json')
meta=read(out/'launcher.json')
if old and old['launcher']['pid']==meta['pid'] and time.time()-old['time']<600:print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
proc=Path(f'/proc/{meta["pid"]}/cmdline')
r={'time':time.time(),'launcher':meta,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,'status':read(out/'status.json'),
 'result_summary':{k:{a:b for a,b in v.items() if a not in ['audit_action_predictions','audit_indices']} for k,v in (read(out/'result.json') or read(out/'progress.json') or {}).get('arms',{}).items()},
 'tail':(out/'driver.log').read_text(errors='replace')[-1600:]}
(out/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
