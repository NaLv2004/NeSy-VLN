import json,time
from pathlib import Path
OUT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diagnostic_relational_rules_v62')
def read(name):
    p=OUT/name;return json.loads(p.read_text()) if p.exists() else {}
launch=read('launcher.json');p=Path(f"/proc/{launch['pid']}/cmdline")
result=dict(time=time.time(),launch=launch,command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else '',
    status=read('status.json'),result=read('result.json'))
log=OUT/'probe.log'
with log.open('rb') as f:
    f.seek(max(0,log.stat().st_size-3000));result['log_tail']=f.read().decode(errors='replace')
(OUT/'last_observation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({**{k:v for k,v in result.items() if k!='result'},
    'result':{k:result['result'].get(k) for k in ('seconds','updates','history','observation_renaming_error')} if result['result'] else {}}))
