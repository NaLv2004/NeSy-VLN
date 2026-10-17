import json,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');out=base/'spatial_automaton_v43'
read=lambda p:json.loads(p.read_text()) if p.exists() else None
old=read(out/'inspection.json')
if old and time.time()-old['time']<600:
    print(json.dumps({'skipped':'minimum600seconds','next_check':old['time']+600}));raise SystemExit
launcher=read(out/'launcher.json');cmd=Path(f'/proc/{launcher["pid"]}/cmdline')
children=read(out/'processes.json');sim=None
if children:
    p=Path(f'/proc/{children["simulator_pid"]}/cmdline');sim=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
record={'time':time.time(),'launcher':launcher,'live':cmd.read_bytes().replace(b'\0',b' ').decode() if cmd.exists() else None,
        'simulator_live':sim,'status':read(out/'status.json'),'commit':read(out/'COMMITTED'),
        'training_metrics':read(out/'training_metrics.json'),'tail':(out/'driver.log').read_text(errors='replace')[-1200:]}
(out/'inspection.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
