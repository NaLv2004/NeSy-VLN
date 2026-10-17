"""One startup-only verification after the initial process check had no update."""
import json,time
from pathlib import Path
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'full_automaton_rl_v62';target=out/'startup_inspection.json'
assert not target.exists(), 'Use the normal minimum600s inspection after startup'
old=json.loads((out/'inspection.json').read_text());assert all(j['latest_commit'] is None for j in old['jobs'].values())
jobs={}
for e in json.loads((out/'launcher.json').read_text())['jobs']:
 folder=out/e['arm'];s=json.loads((folder/'supervisor.json').read_text());p=Path(f'/proc/{s["pid"]}/cmdline')
 status=folder/'status.json';marker=folder/'latest/COMMITTED'
 jobs[e['arm']]={'supervisor':s,'trainer_live':p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None,
  'status':json.loads(status.read_text()) if status.exists() else None,'commit':json.loads(marker.read_text()) if marker.exists() else None,
  'tail':(folder/'driver.log').read_text(errors='replace')[-1800:]}
r={'time':time.time(),'jobs':jobs};target.write_text(json.dumps(r,indent=2))
for j in jobs.values():
 s=j['status'] or {}
 if isinstance(s.get('training'),dict):s['training']={k:v for k,v in s['training'].items() if k!='groups'}
 if isinstance(s.get('development'),dict):s['development']={k:v for k,v in s['development'].items() if k!='groups'}
 if s.get('state')!='failed':j.pop('tail',None)
print(json.dumps(r,indent=2))
