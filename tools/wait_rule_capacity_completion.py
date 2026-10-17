import json,os,select,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/paired_goal_probe_v12')
record=json.loads((out/'rule_capacity_process.json').read_text());pid=record['pid'];p=Path(f'/proc/{pid}/cmdline')
cmd=p.read_bytes() if p.exists() else b'';terminal=not cmd
if cmd:
    assert b'probe_discrete_rule_capacity.py' in cmd;fd=os.pidfd_open(pid)
    try:terminal=bool(select.select([fd],[],[],600)[0])
    finally:os.close(fd)
result=json.loads((out/'rule_capacity.json').read_text()) if (out/'rule_capacity.json').exists() else None
print(json.dumps({'time':time.time(),'pid':pid,'terminal':terminal,'state':(result or {}).get('state'),
    'results':{k:{'found_actions':v['reachable_actions'],'total':v['actions'],
        'unfound_actions':[i for i,r in enumerate(v['best']) if r['top_action']!=i]}
        for k,v in (result or {}).get('results',{}).items()},
    'interpretation':'A lower bound on actions with found supporting Boolean states; failure to find is not proof of impossibility.',
    'tail':(out/'rule_capacity.log').read_text(errors='replace')[-500:]},indent=2))
