import json,shutil,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/contextual_component_v71');read=lambda p:json.loads(p.read_text()) if p.exists() else None
last=read(b/'inspection.json')
if last and time.time()-last['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
jobs=[]
for j in read(b/'launcher.json')['jobs']:
 folder=b/j['arm'];p=Path(f'/proc/{j["pid"]}/cmdline');s=read(folder/'status.json')
 entry=dict(j,live=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,status=s,
  result=read(folder/'result.json'),best_commit=read(folder/'best_audit/COMMITTED'))
 if not s or s.get('state')=='failed':entry['tail']=(folder/'driver.log').read_text(errors='replace')[-2000:]
 jobs.append(entry)
r={'time':time.time(),'jobs':jobs,'free_gib':shutil.disk_usage(b).free/2**30}
(b/'inspection.json').write_text(json.dumps(r,indent=2))
for entry in r['jobs']:
 if entry['result']:
  result=entry['result'];entry['result']={'best_audit_accuracy':result['best_audit_accuracy'],'finished':result['finished'],'measurements':[]}
  for m in result['measurements']:
   compact={'update':m['update'],'fit_accuracy':m['fit']['accuracy'],'audit_accuracy':m['audit']['accuracy'],'seconds':m['seconds']}
   for key in ['audit_text_erased','audit_text_shuffled']:
    if key in m:
     compact[key]=m[key]['accuracy']
     compact[key+'_changed_actions']=sum(a!=b for a,b in zip(m['audit']['predictions'],m[key]['predictions']))
   entry['result']['measurements'].append(compact)
print(json.dumps(r,indent=2))
