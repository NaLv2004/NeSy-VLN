import json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010/ppo_update_strength_v89')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
launch=read(b/'launcher.json');p=Path(f'/proc/{launch["pid"]}/cmdline')
r=dict(time=time.time(),launcher=launch,live=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
 status=read(b/'status.json'),result=read(b/'result.json'))
if not r['result']:r['log_tail']=(b/'driver.log').read_text(errors='replace')[-4000:]
(b/'inspection.json').write_text(json.dumps(r,indent=2))
if r['result']:r['result']={k:v for k,v in r['result'].items() if k in ['preflight','finished','maximum_probability_error','maximum_observed_id_mismatch']}
print(json.dumps(r,indent=2))
