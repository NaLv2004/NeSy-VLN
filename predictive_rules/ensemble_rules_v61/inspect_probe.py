"""Inspect only this CPU pilot and its exact process identity."""
import json,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');OUT=ROOT/'diagnostic_ensemble_rules_v61'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
launch=read(OUT/'launcher.json');pid=launch.get('pid');p=Path(f'/proc/{pid}/cmdline')
r=dict(time=time.time(),launcher=launch,status=read(OUT/'status.json'),
    command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else '',
    log_tail=(OUT/'probe.log').read_text(errors='replace').splitlines()[-4:] if (OUT/'probe.log').exists() else [],result_exists=(OUT/'result.json').exists())
(OUT/'last_observation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
