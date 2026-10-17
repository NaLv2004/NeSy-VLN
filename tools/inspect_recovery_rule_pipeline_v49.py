import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/recovery_rule_learning_v49');read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json')
if old and time.time()-old['time']<600:print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
meta=read(out/'launcher.json');state=read(out/'pipeline_status.json')
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for j in (state or {}).get('jobs',[]):
 folder=out/j['arm'];jobs[j['arm']]={'pid':j['pid'],'live':live(j['pid']),'status':read(folder/'status.json'),'commit':read(folder/'COMMITTED'),'summary':read(folder/'development/summary.json'),'tail':(folder/'driver.log').read_text(errors='replace')[-1800:]}
r={'time':time.time(),'launcher':meta,'live':live(meta['pid']),'pipeline':state,'jobs':jobs,'data_commit':read(out.parent/'data/onpolicy_recovery_v48/COMPLETE'),'encoding':read(out.parent/'data/onpolicy_recovery_v48/encoding_status.json'),'tail':(out/'pipeline.log').read_text(errors='replace')[-2200:]};(out/'inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
