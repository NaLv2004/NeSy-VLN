import json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/perceptual_sequences_v73');read=lambda p:json.loads(p.read_text()) if p.exists() else None
last=read(b/'inspection.json')
if last and time.time()-last['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
jobs=[]
for j in read(b/'launcher.json')['jobs']:
 folder=b/j['arm'];p=Path(f'/proc/{j["pid"]}/cmdline');s=read(folder/'status.json')
 entry=dict(j,live=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,status=s,commit=read(folder/'TOKENS_COMMITTED'))
 if not s or s.get('state')=='failed':entry['tail']=(folder/'driver.log').read_text(errors='replace')[-2300:]
 jobs.append(entry)
r={'time':time.time(),'jobs':jobs};(b/'inspection.json').write_text(json.dumps(r,indent=2))
for entry in r['jobs']:
 if entry['commit']:entry['commit']={k:entry['commit'][k] for k in ['rows','episodes','old_count','canonical_visual_batch','tokens_sha256','training_rows_sha256','seconds']}
print(json.dumps(r,indent=2))
