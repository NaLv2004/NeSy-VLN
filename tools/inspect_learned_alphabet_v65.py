import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/learned_alphabet_v65')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json');meta=read(out/'launcher.json')
if old and time.time()-old['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
proc=Path(f'/proc/{meta["pid"]}/cmdline')
r={'time':time.time(),'launcher':meta,'live':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
 'status':read(out/'status.json'),'result':read(out/'result.json'),
 'tail':(out/'driver.log').read_text(errors='replace')[-1500:]}
(out/'inspection.json').write_text(json.dumps(r,indent=2))
if r['result']:
 r['result']={'initial_code_agreement_with_saved_v53':r['result'].get('initial_code_agreement_with_saved_v53'),
  'finished':r['result']['finished'],'arms':{a:{'best':v['best_audit_accuracy'],'final':{k:z for k,z in v['measurements'][-1]['audit'].items() if k!='predictions'}} for a,v in r['result']['arms'].items()}}
if (r['status'] or {}).get('state')!='failed':r.pop('tail',None)
print(json.dumps(r,indent=2))
