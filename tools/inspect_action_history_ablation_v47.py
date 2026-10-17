import json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010/action_history_ablation_v47');read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(base/'inspection.json')
if old and time.time()-old['time']<600:print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
def live(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
jobs={}
for arm in ['without_previous_action','action_dropout']:
 out=base/arm;meta=read(out/'launcher.json');children=read(out/'processes.json')
 jobs[arm]={'launcher':meta,'live':live(meta['pid']),'status':read(out/'status.json'),'commit':read(out/'COMMITTED'),'training_metrics':read(out/'training_metrics.json'),'summary':read(out/'development/summary.json'),'simulators':{str(pid):live(pid) for pid in (children or {}).get('simulator_pids',[])},'tail':(out/'driver.log').read_text(errors='replace')[-1600:]}
result={'time':time.time(),'jobs':jobs};(base/'inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
