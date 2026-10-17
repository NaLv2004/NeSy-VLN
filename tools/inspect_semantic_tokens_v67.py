import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/semantic_tokens_v67');read=lambda p:json.loads(p.read_text()) if p.exists() else None
meta=read(out/'pilot_launcher.json');old=read(out/'pilot_inspection.json')
if old and time.time()-old['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
p=Path(f'/proc/{meta["pid"]}/cmdline');result=read(out/'pilot_result.json')
r={'time':time.time(),'launcher':meta,'live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
 'status':read(out/'pilot_status.json'),'result':result,'tail':(out/'pilot.log').read_text(errors='replace')[-1600:]}
(out/'pilot_inspection.json').write_text(json.dumps(r,indent=2))
if result:r['result']={'seconds':result['seconds'],'answers':[{'image':a['row']['path'],'caption':a['caption'],'terminated':a['terminated']} for a in result['answers']]}
if (r['status'] or {}).get('state')!='failed':r.pop('tail',None)
print(json.dumps(r,indent=2))
