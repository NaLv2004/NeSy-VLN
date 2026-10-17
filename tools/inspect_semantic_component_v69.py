import json,shutil,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/semantic_component_v69');read=lambda p:json.loads(p.read_text()) if p.exists() else None
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
r={'time':time.time(),'jobs':jobs,'free_gib':shutil.disk_usage(b).free/2**30,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
(b/'inspection.json').write_text(json.dumps(r,indent=2))
for entry in r['jobs']:
 if entry['result']:
  entry['result'].pop('manifest',None)
  for m in entry['result']['measurements']:
   for v in m.values():
    if isinstance(v,dict):v.pop('predictions',None)
print(json.dumps(r,indent=2))
