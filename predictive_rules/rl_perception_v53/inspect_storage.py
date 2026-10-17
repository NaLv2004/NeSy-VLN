import json,os,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
names=('run_raw_grounding_v31','run_rule_planning_v32','run_action_grounded_v33')
results={}
for name in names:
    folder=ROOT/name
    results[name]=dict(checkpoints=[dict(name=p.name,bytes=p.stat().st_size,inode=p.stat().st_ino) for p in folder.glob('checkpoint_*.pt')],
        metadata={p.name:json.loads(p.read_text()) for p in folder.glob('*.json') if p.name in
                  ('status.json','supervisor.json','best.COMMITTED.json','latest.COMMITTED.json')})
processes=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:text=p.read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if str(ROOT) in text and any(name in text for name in names):processes.append(dict(pid=p.parent.name,cmd=text))
print(json.dumps(dict(time=time.time(),runs=results,processes=processes)))
