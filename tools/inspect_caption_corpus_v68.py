import json,subprocess,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/semantic_tokens_v68/corpus')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
last=read(b/'inspection.json')
if last and time.time()-last['time']<600:
 print(json.dumps({'skipped':'minimum600seconds','next_check':last['time']+600}));raise SystemExit
jobs=[]
for j in read(b/'launcher.json')['jobs']:
 p=Path(f'/proc/{j["pid"]}/cmdline');s=read(b/f'rank{j["rank"]}.status.json')
 entry=dict(j,live=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,status=s,result=read(b/f'rank{j["rank"]}.result.json'))
 if not s or s.get('state')=='failed':entry['tail']=(b/f'rank{j["rank"]}.log').read_text(errors='replace')[-1800:]
 jobs.append(entry)
r={'time':time.time(),'jobs':jobs,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
(b/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
